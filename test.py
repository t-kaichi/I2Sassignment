import os, sys
import numpy as np
from utils import reset_tf, set_seed
from glob import glob

from models import (build_pointnet_transformer_model,
                    build_pointnet_model, build_transformer_model)
from eval_utils import assign_batch, make_confusion_matrix
from IMUsSequence import CMU_MoCap, TotalCapture, IMUsSequence
from absl import app
from absl import flags
import myFlags
FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.test_all is None:
        test(FLAGS.load_model)
    else:
        model_dirs = glob(os.path.join(FLAGS.test_all, "*"))
        for model_dir in model_dirs:
            fn = os.path.join(model_dir, "best.weights.hdf5")
            if os.path.exists(fn):
                test(fn)

def test(MODEL_FN):
    reset_tf(FLAGS.device)
    set_seed()
    print("test " + os.path.dirname(MODEL_FN).split("/")[-1])

    is_full_motion = MODEL_FN.find("full_motion") != -1
    root_joint = os.path.dirname(MODEL_FN).split("-")[-2][5:] if MODEL_FN.find("root_") != -1 else None

    # dataset
    if MODEL_FN.find("CMU_MoCap") != -1:
        ds_name = "CMU_MoCap"
        dataset = CMU_MoCap(root_joint)
        if MODEL_FN.find("full_body") != -1:
            attach = "full_body"
            used_imus = list(range(15))
        elif MODEL_FN.find("lower_body") != -1:
            attach = "lower_body"
            used_imus = [2,9,10,11,12,13,14]
        elif MODEL_FN.find("upper_body") != -1:
            attach = "upper_body"
            used_imus = [0,1,2,3,4,5,6,7,8]
        else:
            raise NotImplementedError("attachment")
        subject_ids = list(range(120, 145)) if is_full_motion else "test_json"
        scenes = None
    elif MODEL_FN.find("TotalCapture") != -1:
        ds_name = "TotalCapture"
        dataset = TotalCapture(root_joint)
        if MODEL_FN.find("full_body") != -1:
            attach = "full_body"
            used_imus = list(range(13))
        elif MODEL_FN.find("lower_body") != -1:
            attach = "lower_body"
            used_imus = [2,7,8,9,10,11,12]
        elif MODEL_FN.find("upper_body") != -1:
            attach = "upper_body"
            used_imus = [0,1,2,3,4,5,6]
        else:
            raise NotImplementedError("attachment")
        subject_ids = [5]
        scenes = ['freestyle','walking','acting','rom'] if is_full_motion else ["walking"]
    label_imus = list(range(len(used_imus)))

    assert len(used_imus) == len(label_imus)

    input_shape = (len(used_imus), dataset.window_size, 6, 1)
    test_gen = IMUsSequence(batch_size=FLAGS.batch_size, dataset=dataset,
            used_imus=used_imus, label_imus=label_imus, subject_ids=subject_ids,
            scenes=scenes, random_state=1) # set random_state when testing

    print("loading model ...")
    if MODEL_FN.find("pointnet_transformer-") != -1:
        model_name = "pointnet_transformer-"
        model = build_pointnet_transformer_model(
                nb_classes=test_gen.nb_classes, input_shape=input_shape)
    elif MODEL_FN.find("pointnet-") != -1:
        model_name = "pointnet-"
        model = build_pointnet_model(test_gen.nb_classes, input_shape=input_shape)
    elif MODEL_FN.find("transformer-") != -1:
        model_name = "transformer-"
        model = build_transformer_model(test_gen.nb_classes, input_shape=input_shape)
    model.load_weights(MODEL_FN)

    preds = model.predict(test_gen)
    gts = np.reshape(test_gen.y[:,1:].argmax(axis=-1), (-1,))
    
    assigned = assign_batch(preds)
    cm = make_confusion_matrix(assigned, gts)
    acc = np.sum(np.diag(cm)) / np.sum(cm)

    print("dataset: ", ds_name[:-1])
    print("model: ", model_name)
    print("is_full_motion: ", is_full_motion)
    print("imu attach: ", attach)
    print("root segment: ", root_joint)
    print('accuracy: {}'.format(acc))
    ## confusion matrix
    fn_pfx = model_name+"_"+ds_name+"_"+attach+"_"+root_joint
    return fn_pfx, acc

if __name__ == "__main__":
    app.run(main)