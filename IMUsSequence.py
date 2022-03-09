import copy
import os, sys
import random
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from glob import glob
from absl import flags
from tqdm import tqdm

from split import shuffle_arrays, shuffle_imu
from utils import gyro_conv
import myFlags
FLAGS = flags.FLAGS

class TotalCapture():
    Labels =['Head','Sternum','Pelvis','L_UpArm','R_UpArm',\
        'L_LowArm','R_LowArm','L_UpLeg','R_UpLeg','L_LowLeg',\
        'R_LowLeg','L_Foot','R_Foot']
    DATASET_ROOT_PATH = FLAGS.datapath
    ObjectCategories = len(Labels)
    SymmeLabels = ['Head','Sternum','Pelvis','UpArm','LowArm',\
        'UpLeg','LowLeg','Foot']
    window_size = 120
    window_interval = 15
    ignore_frames = 240 # remove T-pose frames
    
    def __init__(self, root=None, used_imus=list(range(13)), label_imus=list(range(13))):
        self.used_imus = copy.deepcopy(used_imus)
        self.label_imus = copy.deepcopy(label_imus)
        self.nb_classes = len(set(label_imus))
        self.root_joint = "Pelvis" if root is None else root
    
    def path(self, data_type=-1, subject_id=-1, scene=-1):
        p = self.DATASET_ROOT_PATH

        # data_type is 'orientation' or 'acceleration'
        if data_type == -1:
            return p
        p0 = os.path.join(p, data_type)

        if subject_id == -1:
            return p
        scene_name = 's'+str(subject_id)
        p = os.path.join(p0, scene_name)

        if scene == -1:
            return p
        scene_name = scene_name + '_' + scene + '.npy'
        p = os.path.join(p0, scene_name)
        return p
    
    def files_to_load(self, data_type, subject_ids, scenes):
        # data_type is 'orientation' or 'acceleration'
        p = os.path.join(self.DATASET_ROOT_PATH, data_type)
        dst = []
        for ins_id in subject_ids:
            for scene in scenes:
                fpath = glob(p+'/s{}_{}*.npy'.format(ins_id, scene))
                dst.extend(fpath)
        return dst

    def get_class_id(self, sub):
        return self.Labels.index(sub)
         
    def get_class_name(self, sub):
        if self.nb_classes == 8:
            return self.SymmeLabels[int(sub)]
        else:
            return self.Labels[int(sub)]

    def get_class_names(self, arr):
        return np.array([self.get_class_name(elem) for elem in arr])
    
    def get_different_modal_path(self, fpath, modal):
        oripath = os.path.dirname(fpath).split('/')[:-1] + [modal, os.path.basename(fpath)]
        return ''.join([st + '/' for st in oripath])[:-1]

class CMU_MoCap():
    Labels = ["head","thorax","lowerback","lhumerus","rhumerus",
              "lradius","rradius","lwrist","rwrist","lfemur",
              "rfemur","ltibia","rtibia","lfoot","rfoot"]
    DATASET_ROOT_PATH = FLAGS.datapath
    ObjectCategories = len(Labels)
    window_size = 240
    window_interval = 30
    ignore_frames = 1
    
    def __init__(self, root=None, used_imus=list(range(13)), label_imus=list(range(13))):
        self.used_imus = copy.deepcopy(used_imus)
        self.label_imus = copy.deepcopy(label_imus)
        self.nb_classes = len(set(label_imus))
        self.root_joint = "lowerback" if root is None else root
    
    def path(self, data_type=-1, subject_id=-1, scene=-1):
        p = self.DATASET_ROOT_PATH

        # data_type is 'orientation', 'acceleration', or "gyro"
        if data_type == -1:
            return p
        p0 = os.path.join(p, data_type)

        if subject_id == -1:
            return p
        scene_name = 's'+str(subject_id)
        p = os.path.join(p0, scene_name)

        if scene == -1:
            return p
        scene_name = scene_name + '_' + scene + '.npy'
        p = os.path.join(p0, scene_name)
        return p
    
    def files_to_load(self, data_type, subject_ids=None, scenes=None):
        # data_type is 'orientation', 'acceleration', or "gyro"
        p = os.path.join(self.DATASET_ROOT_PATH, data_type)
        dst = []
        if subject_ids in ["train_json", "valid_json", "test_json"]:
            dtype = subject_ids[:-5]
            #with open(FLAGS.scene_file, "r") as rf:
            with open(os.path.join(self.DATASET_ROOT_PATH, "walk.json"), "r") as rf:
                files = json.load(rf)
                for subj in files[dtype]:
                    dst.extend([p + "/s{:03}_{:02}.npy".format(int(subj), sce) \
                        for sce in files[dtype][subj]])
        else:
            if subject_ids is None and scenes is None:
                dst = glob(p+"/s*.npy")
            elif subject_ids is None and scenes is not None:
                for scene in scenes:
                    if type(scene) != str:
                        scene = str(scene).zfill(2)
                    fpath = glob(p+"/s*_{}.npy".format(scene))
                    dst.extend(fpath)
            elif subject_ids is not None and scenes is None:
                for ins_id in subject_ids:
                    if type(ins_id) != str:
                        ins_id = str(ins_id).zfill(3)
                    fpath = glob(p+"/s{}_*.npy".format(ins_id))
                    dst.extend(fpath)
            elif subject_ids is not None and scenes is not None:
                for ins_id in subject_ids:
                    if type(ins_id) != str:
                        ins_id = str(ins_id).zfill(3)
                    for scene in scenes:
                        if type(scene) != str:
                            scene = str(scene).zfill(2)
                        fpath = p+'/s{}_{}.npy'.format(ins_id, scene)
                        if os.path.exists(fpath):
                            dst.append(fpath)
            else:
                raise NotImplementedError("files_to_load error")
        return dst

    def get_class_id(self, sub):
        return self.Labels.index(sub)
         
    def get_class_name(self, sub):
        return self.Labels[int(sub)]

    def get_class_names(self, arr):
        return np.array([self.get_class_name(elem) for elem in arr])
    
    def get_different_modal_path(self, fpath, modal):
        oripath = os.path.dirname(fpath).split('/')[:-1] + [modal, os.path.basename(fpath)]
        return ''.join([st + '/' for st in oripath])[:-1]

class IMUsSequence(Sequence):
    def __init__(self, batch_size, dataset,
                 data_types = ['acceleration','gyro'],
                 used_imus = list(range(13)),
                 label_imus = list(range(13)),
                 random_state=None,
                 dims= 6, # sum of input dimensions
                 subject_ids=[1,2,3], # 4 for val, 5 for test
                 scenes=['freestyle','walking','acting'],
                 is_acc_norm = True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_types = data_types
        self.used_imus = used_imus
        self.label_imus = label_imus
        self.is_acc_norm = is_acc_norm
        self.window_size = dataset.window_size 
        self.window_interval = dataset.window_interval
        self.ignore_frames = dataset.ignore_frames
        self.random_state = random_state
        self.dims = dims
        self.subject_ids = copy.deepcopy(subject_ids)
        self.scenes = copy.deepcopy(scenes)

        self.nb_imus = len(self.used_imus)
        # root joint is ignored in target and label
        self.root_id = self.dataset.get_class_id(self.dataset.root_joint)
        self.nb_classes = self.get_nb_classes()
        self.nb_target_imus = len(self.used_imus) - 1

        nb_frames = []
        root_rots = []
        # load all imu data
        for dt_count, data_type in enumerate(self.data_types):
            files_to_load = self.dataset.files_to_load(data_type, self.subject_ids, self.scenes)
            files_to_load.sort()
            xs = []
            for fpath in tqdm(files_to_load, desc=data_type):
                sample = np.expand_dims(np.load(fpath), axis=-1)# (imu_id, nb_frames, quaternion, 1)
                xs.append(sample)
                if dt_count == 0:
                    nb_frames.append(sample.shape[1])
                    rot_fpath = self.dataset.get_different_modal_path(fpath, 'orientation')
                    root_rot = np.load(rot_fpath)[self.root_id] #(nb_frames, 3, 3) rot_mat
                    root_rots.append(root_rot)
            xs = np.concatenate(xs, axis=1)
            if dt_count == 0:
                stock_x = xs
            else:
                stock_x = np.concatenate([stock_x, xs], axis=2) # (nb_all_imus, frame, sum_datadims, 1)
        #root_rots = np.concatenate(xs, axis=0) 
        root_rots = np.concatenate(root_rots, axis=0)
        self.nb_frames = np.asarray(nb_frames, dtype='uint16')
        if self.dataset.DATASET_ROOT_PATH.find("TotalCapture") != -1:
            self.window_first_ids =self.get_window_first_ids_removeBackWalk(files_to_load) # first frames of windows
        else:
            self.window_first_ids =self.get_window_first_ids() # first frames of windows
        self.nb_samples = len(self.window_first_ids)
        self.redefine_labels() # lift root joint to top of self.used_imus and change labels

        self.x = np.empty((self.nb_samples,) +
                self.get_x_sample_shape(), dtype=self.get_x_dtype())
        self.rot_x = np.empty((self.nb_samples,) + # rotation mat of root joint
                self.get_rot_x_sample_shape(), dtype=self.get_x_dtype())
        self.y = np.empty((self.nb_samples,) +
                self.get_y_sample_shape(), dtype=self.get_y_dtype())

        for frame_i, frame in enumerate(self.window_first_ids):
            for cat_i, cat in enumerate(self.used_imus):
                # only used IMU data is extracted
                self.x[frame_i, cat_i] = stock_x[cat, frame:frame+self.window_size] #(nb_samples, nb_used_imus, sum_datadims, 1)
                self.y[frame_i, cat_i] = self.label_imus[cat_i] #(nb_samples, nb_imus,1)
            self.rot_x[frame_i] = root_rots[frame] #(nb_samples, nb_imus,1)
        self.y = to_categorical(self.y, num_classes=self.nb_classes)
        
        if self.is_acc_norm: # subtract root acceleration to normalize
            self.x[:,:,:,:3] -= self.x[:,[0],:,:3]

        shuffle_arrays([self.x, self.rot_x, self.y]) # shuffle frames
        shuffle_imu([self.x, self.y], ignoreTop=True) # shuffle IMUs

    def get_x_sample_shape(self):
        return (self.nb_imus, self.window_size, self.dims, 1)

    def get_rot_x_sample_shape(self):
        return (3,3)

    def get_y_sample_shape(self):
        return (self.nb_imus,)
    
    def get_rot_y_sample_shape(self):
        return (self.window_size,3,3)

    def get_x_dtype(self):
        return "float32"

    def get_y_dtype(self):
        return "uint8"
        #return "float32"

    def __len__(self):
        return int(np.ceil(float(self.nb_samples) / float(self.batch_size)))

    def actual_len(self):
        # actual length without repeated augumentation
        return int(np.ceil(float(self.nb_samples) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        idx %= self.actual_len()
        start = idx * self.batch_size

        batch_x = self.x[start:start + self.batch_size]
        ys = self.y[start:start + self.batch_size, 1:]
        rot_xs = self.rot_x[start:start + self.batch_size]
        xs = np.stack([self.augment(x, rot_x) for x, rot_x in 
                zip(batch_x.astype(self.get_x_dtype()), rot_xs.astype(self.get_x_dtype()))], axis=0)
        return xs, ys
    
    def augment(self, x, rot_x=None):
        # x[:,:,:3] is acceleration, x[:,:,3:] is gyro
        if rot_x is not None:
            dst = np.empty_like(x)
            dst[:,:,:3] = np.matmul(rot_x.T, x[:,:,:3])
            dst[:,:,3:] = np.matmul(gyro_conv(rot_x.T), x[:,:,3:])
            return dst
        else:
            return x

    def on_epoch_end(self):
        if self.random_state is None:
            shuffle_arrays([self.x, self.rot_x, self.y]) # shuffle frames
            shuffle_imu([self.x, self.y], ignoreTop=True) # shuffle IMUs
            pass
    
    def get_window_first_ids(self):
        prev_frames_sum = 0
        dst = []
        for nb_frame in self.nb_frames:
            if nb_frame-self.ignore_frames*2 < self.window_size:
                #raise ValueError('number of frames is small')
                pass
            for idx in range(self.ignore_frames, nb_frame-self.ignore_frames-self.window_size, self.window_interval):
                dst.append(prev_frames_sum + idx)
            prev_frames_sum += nb_frame
        random.shuffle(dst)
        return dst

    def get_window_first_ids_removeBackWalk(self, files_to_load):
        assert len(self.nb_frames) == len(files_to_load)
        #remove_fn, remove_frame
        prev_frames_sum = 0
        tmp = []
        for nb_frame, fn in zip(self.nb_frames, files_to_load):
            if nb_frame-self.ignore_frames*2 < self.window_size:
                #raise ValueError('number of frames is small')
                pass
            for idx in range(self.ignore_frames, nb_frame-self.ignore_frames-self.window_size, self.window_interval):
                tmp.append(prev_frames_sum + idx)
            prev_frames_sum += nb_frame

        with open(os.path.join(self.dataset.DATASET_ROOT_PATH, "backwalk.json"), 'r') as f:
            remove_files = json.load(f) # {"fn": [[frame_start, frame_end],[,]..]}
        tmp_frames = 0
        remove_frames = []
        for nb_frame, fn in zip(self.nb_frames, files_to_load):
            for rm_fn in remove_files.keys():
                if fn.find(rm_fn) != -1:
                    rm_frames = remove_files[rm_fn] #[[frame_start, frame_end], [,]..]
                    for rm_frame in rm_frames:
                        rm_start = tmp_frames + rm_frame[0]
                        rm_end = tmp_frames + rm_frame[1]
                        remove_frames.append([rm_start, rm_end])
            tmp_frames += nb_frame

        dst = []
        for frame in tmp:
            should_remain = True
            for remove_frame in remove_frames:
                if frame + self.window_size > remove_frame[0] and frame < remove_frame[1]:
                    should_remain = False
            if should_remain:
                dst.append(frame)
        random.shuffle(dst)
        return dst

    def redefine_labels(self):
        root_idx = self.used_imus.index(self.root_id)
        self.used_imus.remove(self.root_id)
        self.used_imus = [self.root_id] + self.used_imus
        
        root_label = self.label_imus[root_idx]
        self.label_imus.remove(root_label)
        self.label_imus = np.array(self.label_imus)
        self.label_imus = [0] + list(np.where(self.label_imus > root_label,
                    self.label_imus-1, self.label_imus))

    def get_nb_classes(self):
        tmp = copy.deepcopy(self.used_imus)
        tmp.remove(self.root_id) 
        return len(set(tmp))