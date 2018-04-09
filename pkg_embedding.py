import argparse
import os
import sys

import models
import config
from pkg.pkg import PKG


def parse_args():
    """
    parse command line input
    """
    parser = argparse.ArgumentParser(description="Train personal knowledge embeddings.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-openkg", action="store", dest="openkg_dir", required=True,
                        help="directory of openkg graph")
    parser.add_argument("-pkg", action="store", dest="pkg_dir", required=True,
                        help="directory of pkg dir")
    parser.add_argument("-output", action="store", dest="output_dir", required=True,
                        help="directory of output dir")
    parser.add_argument("-model", action="store", dest="model_name", default="TransE",
                        help="name of the model to train the embeddings, could be TransE, TransH, HolE, RESCAL, etc.")

    options = parser.parse_args()
    # print options
    return options


def convert_pkg_to_openkg(opts):
    pkg = PKG(pkg_path=os.path.join(opts.pkg_dir, "user_app_day_duration.txt"),
              user_list_path=os.path.join(opts.pkg_dir, "user_id_imei_birth_gender.txt"),
              app_list_path=os.path.join(opts.pkg_dir, "app_id_package_usercount.txt"))
    print("PKG loaded.")
    pkg.output_as_openkg(output_dir=opts.openkg_dir, days=[0])


def train_embeddings(opts):
    os.makedirs(opts.output_dir, exist_ok=True)

    con = config.Config()
    con.set_in_path(opts.openkg_dir)

    con.set_test_flag(False)
    con.set_work_threads(4)
    con.set_train_times(500)
    con.set_nbatches(100)
    con.set_alpha(0.001)
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
        con.set_model(models.TransE)
    elif opts.model_name == "TransD":
        con.set_model(models.TransD)
    elif opts.model_name == "TransR":
        con.set_model(models.TransR)
    elif opts.model_name == "TransH":
        con.set_model(models.TransH)
    elif opts.model_name == "ComplEx":
        con.set_model(models.ComplEx)
    elif opts.model_name == "DistMult":
        con.set_model(models.DistMult)
    elif opts.model_name == "HolE":
        con.set_model(models.HolE)
    elif opts.model_name == "RESCAL":
        con.set_model(models.RESCAL)
    else:
        print("Unknown model: " + opts.model_name)
        sys.exit(1)

    # Train the model.
    con.run()
    # To test models after training needs "set_test_flag(True)".
    # con.test()


def evaluate_embeddings(opts):
    PKG.evaluate_embeddings(embedding_path=os.path.join(opts.output_dir, "embedding.vec.json"),
                            openke_dir=opts.openkg_dir,
                            user_info_path=os.path.join(opts.pkg_dir, "user_id_imei_birth_gender.txt"))

def main():
    """
    the main function
    """
    opts = parse_args()

    convert_pkg_to_openkg(opts)
    train_embeddings(opts)
    evaluate_embeddings(opts)

    return


if __name__ == "__main__":
    main()
