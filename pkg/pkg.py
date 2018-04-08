import networkx

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

    def output_as_openkg(self, output_dir, days=None, weight_threshold=0):
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
                included_triples.add((s, t, e_attr["day"]))
        print("{} matched triples".format(len(included_triples)))

        included_nodes = sorted(set([t[0] for t in included_triples] + [t[1] for t in included_triples]))
        included_days = sorted(set([t[2] for t in included_triples]))
        entity2id = {k: v for v, k in enumerate(included_nodes)}
        relation2id = {k: v for v, k in enumerate(included_days)}

        with open(output_dir + "/entity2id.txt", "w") as entity2id_file:
            entity2id_file.write("{}\n".format(len(included_nodes)))
            for entity_id, entity_name in enumerate(included_nodes):
                entity2id_file.write("{} {}\n".format(entity_name, entity_id))
            entity2id_file.close()

        with open(output_dir + "/relation2id.txt", "w") as relation2id_file:
            relation2id_file.write("{}\n".format(len(included_days)))
            for relation_id, relation_name in enumerate(included_days):
                relation2id_file.write("{} {}\n".format(relation_name, relation_id))
            relation2id_file.close()

        with open(output_dir + "/train2id.txt", "w") as train2id_file:
            train2id_file.write("{}\n".format(len(included_triples)))
            for s, t, day in included_triples:
                train2id_file.write("{} {} {}\n".format(entity2id[s], entity2id[t], relation2id[day]))
            train2id_file.close()

    @staticmethod
    def evaluate_embeddings(embedding_path, openke_dir, user_info_path):
        import json
        embedding_file = open(embedding_path, "r")
        embedding_json = json.load(embedding_file)
        ent_embeddings = embedding_json["ent_embeddings"]

        entity2id_file = open(openke_dir + "/entity2id.txt")
        entity2id = {}
        for line in entity2id_file.readlines()[1:]:
            words = line.split()
            if len(words) != 2:
                continue
            entity2id[words[0]] = int(words[1])

        entity2info = PKG.parse_user_list(user_info_path)

        data = []
        target = []
        for entity in entity2id:
            if entity not in entity2info:
                continue
            ent_info = entity2info[entity]
            ent_id = entity2id[entity]
            ent_embedding = ent_embeddings[ent_id]
            ent_gender = ent_info[-1]
            data.append(ent_embedding)
            target.append(ent_gender)

        print("Sample data: %s" % data[0])
        print("Sample target: %s" % target[0])

        # Now run cross-validation
        from sklearn.model_selection import cross_val_score
        from sklearn import svm
        clf = svm.SVC(kernel="linear", C=1)
        scores = cross_val_score(clf, data, target, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == "__main__":
    base_dir = "../temp"
    pkg = PKG(pkg_path=base_dir + "/sample_data/user_app_day_duration.txt",
              user_list_path=base_dir + "/sample_data/user_id_imei_birth_gender.txt",
              app_list_path=base_dir + "/sample_data/app_id_package_usercount.txt")
    print("PKG loaded.")
    pkg.output_as_openkg(output_dir=base_dir+"/openke", days=[0])
