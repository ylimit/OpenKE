from config.Config import Config
from models.TransE import TransE
import tensorflow as tf
import numpy as np
import json

# (1) Set import files and OpenKE will automatically load models via tf.Saver().
con = Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_flag(True)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files("./res/model.vec.tf")
con.init()
con.set_model(TransE)
con.test()

# (2) Read model parameters from json files and manually load parameters.
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# f = open("./res/embedding.vec.json", "r")
# content = json.loads(f.read())
# f.close()
# con.set_parameters(content)
# con.test()

# (3) Manually load models via tf.Saver().
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# con.import_variables("./res/model.vec.tf")
# con.test()
