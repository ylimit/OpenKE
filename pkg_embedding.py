import argparse
import os
import sys

from pkg.pkg import PKG


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="Train personal knowledge embeddings.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-openke", action="store", dest="openke_dir", required=True,
                        help="directory of openke graph")
    parser.add_argument("-pkg", action="store", dest="pkg_dir", required=True,
                        help="directory of pkg dir")
    parser.add_argument("-output", action="store", dest="output_dir", required=True,
                        help="directory of output dir")
    parser.add_argument("-model", action="store", dest="model_name", default="TransE",
                        help="model to generate the embeddings, could be 1hot, TransE, TransH, HolE, RESCAL, etc.")
    parser.add_argument("-ndays", action="store", dest="ndays", default=1,
                        help="number of days of knowledge to include")
    parser.add_argument("-weight_threshold", action="store", dest="weight_threshold", default=0,
                        help="weight threshold of knowledge to include")
    parser.add_argument("-day_rel", action="store_true", dest="day_rel", default=False,
                        help="whether to represent days with different relations")
    parser.add_argument("-phases", action="store", dest="phases", default="gen_kg,train,evaluate,visualize",
                        help="phases to run, could be one or more of gen_kg, train, evaluate and visualize.")
    parser.add_argument("-alpha", action="store", dest="alpha", default=0.001,
                        help="hyper parameter: alpha.")
    parser.add_argument("-nbatches", action="store", dest="nbatches", default=100,
                        help="hyper parameter: nbatches.")

    options = parser.parse_args()
    # print options
    return options


def convert_pkg_to_openke(opts):
    pkg = PKG(pkg_path=os.path.join(opts.pkg_dir, "user_app_day_duration.txt"),
              user_list_path=os.path.join(opts.pkg_dir, "user_id_imei_birth_gender.txt"),
              app_list_path=os.path.join(opts.pkg_dir, "app_id_package_usercount.txt"))
    print("PKG loaded.")
    days = range(0, int(opts.ndays))
    pkg.output_as_openke(output_dir=opts.openke_dir,
                         days=days,
                         day_rel=opts.day_rel,
                         weight_threshold=float(opts.weight_threshold))


def one_hot_embedding(openke_dir):
    train2id_file = open(openke_dir + "/train2id.txt")
    included_triples = []
    for line in train2id_file.readlines()[1:]:
        words = line.split()
        if len(words) != 3:
            continue
        included_triples.append((words[0], words[1], words[2]))

    # Gen one-hot embeddings
    included_objects = sorted(set([t[1] for t in included_triples]))
    object2id = {k: v for v, k in enumerate(included_objects)}
    user2objects = {}
    for s, t, r in included_triples:
        user_id = int(s)
        object_id = object2id[t]
        if user_id not in user2objects:
            user2objects[user_id] = []
        user2objects[user_id].append(object_id)

    embedding_dim = len(included_objects)
    embeddings = {}
    for user_id in user2objects:
        object_ids = user2objects[user_id]
        embedding = [0] * embedding_dim
        for object_id in object_ids:
            embedding[object_id] = 1
        embeddings[user_id] = embedding
    return embeddings


def train_embeddings(opts):
    os.makedirs(opts.output_dir, exist_ok=True)

    if opts.model_name == "1hot":
        import json
        embeddings = one_hot_embedding(opts.openke_dir)
        embedding_file = open(os.path.join(opts.output_dir, "embedding.vec.json"), "w")
        json.dump({"ent_embeddings": embeddings}, embedding_file)
        embedding_file.close()
        print("Successfully generated one-hot embeddings")
        return

    from config.Config import Config
    from models.TransE import TransE
    from models.TransD import TransD
    from models.TransH import TransH
    from models.TransR import TransR
    from models.RESCAL import RESCAL
    from models.DistMult import DistMult
    from models.ComplEx import ComplEx
    from models.HolE import HolE

    con = Config()
    con.set_in_path(opts.openke_dir)

    con.set_test_flag(True)
    con.set_work_threads(4)
    con.set_train_times(500)
    con.set_nbatches(int(opts.nbatches))
    con.set_alpha(float(opts.alpha))
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(64)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adam")

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(os.path.join(opts.output_dir, "model.vec.tf"), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(os.path.join(opts.output_dir, "embedding.vec.json"))
    # Initialize experimental settings.
    con.init()

    # Set the knowledge embedding model
    if opts.model_name == "TransE":
        con.set_model(TransE)
    elif opts.model_name == "TransD":
        con.set_model(TransD)
    elif opts.model_name == "TransR":
        con.set_model(TransR)
    elif opts.model_name == "TransH":
        con.set_model(TransH)
    elif opts.model_name == "ComplEx":
        con.set_model(ComplEx)
    elif opts.model_name == "DistMult":
        con.set_model(DistMult)
    elif opts.model_name == "HolE":
        con.set_model(HolE)
    elif opts.model_name == "RESCAL":
        con.set_model(RESCAL)
    else:
        print("Unknown model: " + opts.model_name)
        sys.exit(1)

    # Train the model.
    con.run()
    # To test models after training needs "set_test_flag(True)".
    con.test()


def evaluate_embeddings(opts):
    PKG.evaluate_embeddings(embedding_path=os.path.join(opts.output_dir, "embedding.vec.json"),
                            openke_dir=opts.openke_dir,
                            user_info_path=os.path.join(opts.pkg_dir, "user_id_imei_birth_gender.txt"))


def visualize_embeddings(opts):
    PKG.visualize_embeddings(embedding_path=os.path.join(opts.output_dir, "embedding.vec.json"),
                             openke_dir=opts.openke_dir,
                             user_info_path=os.path.join(opts.pkg_dir, "user_id_imei_birth_gender.txt"))


def main():
    """
    the main function
    """
    opts = parse_args()

    phases = str(opts.phases).split(",")
    if "gen_kg" in phases:
        convert_pkg_to_openke(opts)
    if "train" in phases:
        train_embeddings(opts)
    if "evaluate" in phases:
        evaluate_embeddings(opts)
    if "visualize" in phases:
        visualize_embeddings(opts)
    return


if __name__ == "__main__":
    main()
