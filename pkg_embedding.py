import config
import models
import tensorflow as tf
import numpy as np

con = config.Config()
con.set_in_path("./temp/openke/")

con.set_test_flag(False)
con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("Adam")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./temp/embedding_TransE/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./temp/embedding_TransE/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()
