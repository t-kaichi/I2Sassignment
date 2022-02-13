import os
import time
import copy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from absl import flags
from absl import app

from models import (build_transformer_model, build_pointnet_model,
                    build_pointnet_transformer_model)
from utils import reset_tf, set_seed, imshow
from IMUsSequence import TotalCapture, CMU_MoCap, IMUsSequence

import myFlags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = flags.FLAGS

def main(argv):
    reset_tf(FLAGS.device)
    set_seed()

    # Dataset
    if FLAGS.dataset == "CMU_MoCap":
        ds_info = CMU_MoCap(FLAGS.root)
        if FLAGS.attachment == "full_body":
            used_imus = list(range(15))
        elif FLAGS.attachment == "lower_body":
            used_imus = [2,9,10,11,12,13,14]
        elif FLAGS.attachment == "upper_body":
            used_imus = [0,1,2,3,4,5,6,7,8]
        elif FLAGS.attachment == "action_recognition":
            used_imus = [0,2,7,8,13,14]
        subject_ids = list(range(90)) if FLAGS.scene_file is None else "train_json"
        scenes = None
        v_subject_ids = list(range(90, 120)) if FLAGS.scene_file is None else "valid_json"
        v_scenes = None
    elif FLAGS.dataset == "TotalCapture":
        ds_info = TotalCapture(FLAGS.root)
        if FLAGS.attachment == "full_body":
            used_imus = list(range(13))
        elif FLAGS.attachment == "lower_body":
            used_imus = [2,7,8,9,10,11,12]
        elif FLAGS.attachment == "upper_body":
            used_imus = [0,1,2,3,4,5,6]
        elif FLAGS.attachment == "action_recognition":
            used_imus = [2,3,4,5,6,11,12]
        subject_ids = [1,2,3]
        scenes = ['freestyle','walking','acting','rom'] if FLAGS.scene_file is None else ["walking"]
        v_subject_ids = [4]
        v_scenes = copy.deepcopy(scenes)
    label_imus = list(range(len(used_imus)))
    v_used_imus = copy.deepcopy(used_imus)
    v_label_imus = copy.deepcopy(label_imus)

    input_shape = (len(used_imus), ds_info.window_size, 6, 1) #(6, 1, 1)
    assert len(used_imus) == len(label_imus)

    print("loading datasets ...")
    train_gen = IMUsSequence(batch_size=FLAGS.batch_size, dataset=ds_info,
            used_imus=used_imus, label_imus=label_imus,
            subject_ids=subject_ids, scenes=scenes)
    valid_gen = IMUsSequence(batch_size=FLAGS.batch_size, dataset=ds_info,
            used_imus=v_used_imus, label_imus=v_label_imus,
            subject_ids=v_subject_ids, scenes=v_scenes)

    #check_ds(train_gen) # im tameshi.py

    print("model building ...")
    if FLAGS.model == "pointnet_transformer":
        model = build_pointnet_transformer_model(
                    nb_classes=train_gen.nb_classes, input_shape=input_shape)
    elif FLAGS.model == "pointnet":
        model = build_pointnet_model(
                    nb_classes=train_gen.nb_classes, input_shape=input_shape)
    elif FLAGS.model == "transformer":
        model = build_transformer_model(
                    nb_classes=train_gen.nb_classes, input_shape=input_shape)
    else:
        raise NotImplementedError("model "+ FLAGS.model + " is not implemented")

    # Expr. params
    experiment_title = FLAGS.model
    experiment_title += "-" + FLAGS.dataset
    experiment_title += "-" + FLAGS.attachment
    experiment_title += "-root_" + ds_info.root_joint
    experiment_title += '-%d' % time.time()
    logdir = os.path.join(FLAGS.log_dir, experiment_title)
    os.makedirs(logdir, exist_ok=True)
    print("logdir: ", logdir)

    checkpoint = ModelCheckpoint(logdir + "/best.weights.hdf5", monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=True, mode='auto', save_freq="epoch")
    early_stopping = EarlyStopping(monitor="val_loss", patience=400)
    callbacks = [checkpoint, early_stopping]
    model.fit(train_gen, epochs=1000, validation_data=valid_gen,
             validation_steps=len(valid_gen), callbacks=callbacks)

if __name__ == "__main__":
    app.run(main)