from absl import flags

flags.DEFINE_integer("device", 0, "visible gpu device")
flags.DEFINE_string("scene_file", "../data/CMU_MoCap/data/global/walk.json", "load files when 'walk' otherwise None")
flags.DEFINE_string("root", None, "root joint, None gives pelvis or lowerback")

# train
flags.DEFINE_string("dataset", "TotalCapture", "TotalCapture or CMU_MoCap")
flags.DEFINE_string("datapath", "./data/TotalCapture/global/", "path to the dataset")
flags.DEFINE_string("attachment", "lower_body", "full_body or lower_body or upper_body")
flags.DEFINE_boolean("acc_norm", False, "subtract root acceleration")
flags.DEFINE_string("model", "pointnet_transformer", "pointnet, transformer, or pointnet_transformer")
flags.DEFINE_integer("imu2vec_kernelsize", 9, "kernel size of CNN")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("mlp_nodes", 256, "base number of the CNN layers")
flags.DEFINE_string("optimizer", "RMSprop", "optimizer for training")
flags.DEFINE_float("learning_rate", 0.001, "optimizer for training")

# transformer
flags.DEFINE_integer("embed_dims", 256, "output dimensions of the attention net")
flags.DEFINE_integer("ff_dims", 768, "output dimensions of the feed forward net")
flags.DEFINE_integer("nb_transformer_layers", 4, "N in Transformer")

# test
flags.DEFINE_string("load_model", "./logs/", "path to the trained model")
flags.DEFINE_string("test_all", None, "all models in dir (if it has best.weights.hdf5")