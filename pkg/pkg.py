import os
import sys
import networkx

class PKG(object):
    def __init__(self, pkg_path, user_list_path, app_list_path):
        self.g = networkx.MultiDiGraph()
        self.user_info = {}
        self.app_info = {}
        self.parse_user_list(user_list_path)
        self.parse_app_list(app_list_path)
        self.parse_pkg(pkg_path)

    def parse_user_list(self, user_list_path):
        with open(user_list_path, "r") as user_list_file:
            user_list_lines = user_list_file.readlines()
            for line in user_list_lines:
                words = line.split()
                if len(words) < 1:
                    continue
                user_id = "user_" + words[0]
                self.user_info[user_id] = words[1:]

    def parse_app_list(self, app_list_path):
        with open(app_list_path, "r") as app_list_file:
            app_list_lines = app_list_file.readlines()
            for line in app_list_lines:
                words = line.split()
                if len(words) < 1:
                    continue
                app_id = "app_" + words[0]
                self.app_info[app_id] = words[1:]

    def parse_pkg(self, pkg_path):
        with open(pkg_path, "r") as pkg_file:
            pkg_lines = pkg_file.readlines()
            for line in pkg_lines:
                words = line.split()
                if len(words) != 4:
                    continue
                user_id = "user_" + words[0]
                app_id = "app_" + words[1]
                day, weight = words[2:4]
                self.g.add_edge(user_id, app_id, day=day, weight=weight)

    def output_as_openkg(self, output_dir, days=None, weight_threshold=0):
        included_triples = set()
        for s, t in set(self.g.edges()):
            for i in self.g[s][t]:
                e_attr = self.g[s][t][i]
                if days and e_attr["day"] not in days:
                    continue
                if e_attr["weight"] < weight_threshold:
                    continue
                included_triples.add((s, t, e_attr["day"]))

        zip_triples = zip(*included_triples)
        included_nodes = sorted(set(zip_triples[0] + zip_triples[1]))
        included_days = sorted(set(zip_triples[2]))
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
                entity2id_file.write("{} {}\n".format(relation_name, relation_id))
            relation2id_file.close()

        with open(output_dir + "/train2id.txt", "w") as train2id_file:
            train2id_file.write("{}\n".format(len(included_triples)))
            for s, t, day in included_triples:
                train2id_file.write("{} {} {}\n".format(entity2id[s], entity2id[t], relation2id[day]))
            train2id_file.close()


if __name__ == "__main__":
    base_dir = "C:\Users\liyc\PycharmProjects\OpenKE\\temp"
    pkg = PKG(pkg_path=base_dir + "\sample_data\user_app_day_duration.txt",
              user_list_path=base_dir + "\sample_data\user_id_imei_birth_gender.txt",
              app_list_path=base_dir + "\sample_data\\app_id_package_usercount.txt")
    pkg.output_as_openkg(output_dir=base_dir+"\output", days=[0])
