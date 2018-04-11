import os
import networkx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class PKG(object):
    def __init__(self, pkg_path, user_list_path, app_list_path):
        self.user_info = PKG.parse_user_list(user_list_path)
        self.app_info = PKG.parse_app_list(app_list_path)
        self.g = PKG.parse_pkg(pkg_path)

    @staticmethod
    def parse_user_list(user_list_path):
        with open(user_list_path, "r") as user_list_file:
            user_info = {}
            user_list_lines = user_list_file.readlines()
            for line in user_list_lines:
                words = line.split()
                if len(words) < 1:
                    continue
                user_id = "user_" + words[0]
                user_info[user_id] = words[1:]
            print("{} users in total".format(len(user_info)))
            return user_info

    @staticmethod
    def parse_app_list(app_list_path):
        with open(app_list_path, "r") as app_list_file:
            app_info = {}
            app_list_lines = app_list_file.readlines()
            for line in app_list_lines:
                words = line.split()
                if len(words) < 1:
                    continue
                app_id = "app_" + words[0]
                app_info[app_id] = words[1:]
            print("{} apps in total".format(len(app_info)))
            return app_info

    @staticmethod
    def parse_pkg(pkg_path):
        with open(pkg_path, "r") as pkg_file:
            g = networkx.MultiDiGraph()
            pkg_lines = pkg_file.readlines()
            for line in pkg_lines:
                words = line.split()
                if len(words) != 4:
                    continue
                user_id = "user_" + words[0]
                app_id = "app_" + words[1]
                day, weight = words[2:4]
                g.add_edge(user_id, app_id, day=int(day), weight=float(weight))
            print("{} edges in total".format(len(g.edges())))
            return g

    def output_as_openke(self, output_dir, days=None, day_rel=False, weight_threshold=0, gen_tests=False):
        print("Trying to output knowledge graph to " + output_dir)
        included_triples = set()
        traversed_edges = set()
        for s, t in self.g.edges():
            if (s, t) in traversed_edges:
                continue
            traversed_edges.add((s, t))
            for i in self.g[s][t]:
                e_attr = self.g[s][t][i]
                if days and e_attr["day"] not in days:
                    continue
                if e_attr["weight"] < weight_threshold:
                    continue
                included_triples.add((s, t, (e_attr["day"] if day_rel else 0)))

        print("{} matched triples".format(len(included_triples)))

        included_entities = sorted(set([t[0] for t in included_triples] + [t[1] for t in included_triples]))
        included_relations = sorted(set([t[2] for t in included_triples]))
        entity2id = {k: v for v, k in enumerate(included_entities)}
        relation2id = {k: v for v, k in enumerate(included_relations)}

        with open(output_dir + "/entity2id.txt", "w") as entity2id_file:
            entity2id_file.write("{}\n".format(len(included_entities)))
            for entity_id, entity_name in enumerate(included_entities):
                entity2id_file.write("{} {}\n".format(entity_name, entity_id))
            entity2id_file.close()

        with open(output_dir + "/relation2id.txt", "w") as relation2id_file:
            relation2id_file.write("{}\n".format(len(included_relations)))
            for relation_id, relation_name in enumerate(included_relations):
                relation2id_file.write("{} {}\n".format(relation_name, relation_id))
            relation2id_file.close()

        import random
        included_triples = list(included_triples)
        random.shuffle(included_triples)

        if gen_tests:
            test_len = int(0.1 * len(included_triples))
            valid_len = int(0.2 * len(included_triples))
        else:
            test_len = 0
            valid_len = 0

        with open(output_dir + "/train2id.txt", "w") as train2id_file:
            train_triples = included_triples[valid_len:]
            train2id_file.write("{}\n".format(len(train_triples)))
            for s, t, r in train_triples:
                train2id_file.write("{} {} {}\n".format(entity2id[s], entity2id[t], relation2id[r]))
            train2id_file.close()

        with open(output_dir + "/valid2id.txt", "w") as valid2id_file:
            valid_triples = included_triples[test_len:valid_len]
            valid2id_file.write("{}\n".format(len(valid_triples)))
            for s, t, r in valid_triples:
                valid2id_file.write("{} {} {}\n".format(entity2id[s], entity2id[t], relation2id[r]))
            valid2id_file.close()

        with open(output_dir + "/test2id.txt", "w") as test2id_file:
            test_triples = included_triples[:test_len]
            test2id_file.write("{}\n".format(len(test_triples)))
            for s, t, r in test_triples:
                test2id_file.write("{} {} {}\n".format(entity2id[s], entity2id[t], relation2id[r]))
            test2id_file.close()

    @staticmethod
    def load_embeddings(embedding_path, openke_dir, user_info_path, embedding_format="openke"):
        if embedding_format == "openke":
            import json
            embedding_file = open(embedding_path, "r")
            embedding_json = json.load(embedding_file)
            ent_embeddings = embedding_json["ent_embeddings"]
        elif embedding_format == "openne":
            embedding_file = open(embedding_path, "r")
            ent_embeddings = {}
            for line in embedding_file.readlines()[1:]:
                words = line.split()
                ent_id = int(words[0])
                ent_embedding = [float(word) for word in words]
                ent_embeddings[ent_id] = ent_embedding
        else:
            print("Unknown embedding format: %s" % embedding_format)
            return

        entity2id_file = open(openke_dir + "/entity2id.txt")
        entity2id = {}
        for line in entity2id_file.readlines()[1:]:
            words = line.split()
            if len(words) != 2:
                continue
            entity2id[words[0]] = int(words[1])

        entity2info = PKG.parse_user_list(user_info_path)

        embeddings = []
        genders = []
        ages = []
        for entity in entity2id:
            if entity not in entity2info:
                continue
            ent_info = entity2info[entity]
            ent_id = entity2id[entity]
            ent_embedding = ent_embeddings[ent_id]
            ent_gender = int(ent_info[-1])
            ent_age = 2018 - int(ent_info[-2][:4])
            embeddings.append(ent_embedding)
            genders.append(ent_gender)
            ages.append(ent_age)

        print("Sample embeddings: %s" % embeddings[:5])
        print("Sample genders: %s" % genders[:5])
        print("Sample ages: %s" % ages[:5])
        return embeddings, genders, ages

    @staticmethod
    def evaluate_embeddings(embedding_path, openke_dir, user_info_path, embedding_format="openke"):
        embeddings, genders, ages = PKG.load_embeddings(embedding_path, openke_dir, user_info_path)
        embeddings = np.array(embeddings)
        genders = np.array(genders)

        # balance males and females
        female_idx = np.where(genders == 2)[0]
        male_idx = np.where(genders == 1)[0]
        male_idx = np.random.choice(male_idx, len(female_idx), replace=False)
        user_idx = np.concatenate((female_idx, male_idx))
        embeddings = embeddings[user_idx]
        genders = genders[user_idx]

        # Now run cross-validation
        from sklearn.model_selection import cross_val_score
        from sklearn import svm
        clf = svm.SVC()
        scores = cross_val_score(clf, embeddings, genders, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    @staticmethod
    def visualize_embeddings(embedding_path, openke_dir, user_info_path, embedding_format="openke"):
        embeddings, genders, ages = PKG.load_embeddings(embedding_path, openke_dir, user_info_path)
        print("Computing t-SNE")
        from sklearn.manifold import TSNE
        X_tsne = TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = X_tsne.fit_transform(embeddings)

        print("Visualizing t-SNE")
        PKG.plot_tsne(X_tsne, genders, "t-SNE of the embeddings")
        plt.imsave(os.path.join(os.path.dirname(embedding_path), "t-SNE_user_embedding_gender.png"))

    @staticmethod
    def plot_tsne(X_tsne, targets, title):
        # Scale and visualize the embedding vectors
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X = (X_tsne - x_min) / (x_max - x_min)

        y_min, y_max = np.min(targets), np.max(targets)
        y = (np.array(targets) - y_min) / (y_max - y_min)

        plt.figure(figsize=(10, 10))
        # ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(targets[i]),
                     color=plt.cm.spring(y[i]),
                     fontdict={'weight': 'bold', 'size': 8})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)


if __name__ == "__main__":
    base_dir = "../temp"
    pkg = PKG(pkg_path=base_dir + "/sample_data/user_app_day_duration.txt",
              user_list_path=base_dir + "/sample_data/user_id_imei_birth_gender.txt",
              app_list_path=base_dir + "/sample_data/app_id_package_usercount.txt")
    print("PKG loaded.")
    pkg.output_as_openke(output_dir=base_dir+"/openke", days=[0])
