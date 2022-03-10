def set_seed():
    import numpy as np
    import tensorflow as tf
    import random
    # Fix random seeds
    SEED = 2020
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)


def reset_tf(gpu_id=0):
    # Setup tf
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) < 1:
        print("trained with CPU")
        return 0
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    print("set memory growth GPU No.: {}".format(gpu_id))
    return 0 

def imshow(title, img):
    import cv2
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

def gyro_conv(R):
    import numpy as np

    # isRotationMatrix :
    shouldBeIdentity = np.dot(R.T, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n > 1e-6:
        raise ValueError("R in gyro_conv is not rotation mat")
    
    # mat2euler
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if  not singular :
        r = np.arctan2(R[2,1] , R[2,2])
        p = np.arctan2(-R[2,0], sy)
    else :
        r = np.arctan2(-R[1,2], R[1,1])
        p = np.arctan2(-R[2,0], sy)
 
    # gyro conversion mat
    l2g = np.array([[1, np.sin(r)*np.tan(p), np.cos(r)*np.tan(p)],
                    [0,           np.cos(r),          -np.sin(r)],
                    [0, np.sin(r)/np.cos(p), np.cos(r)/np.cos(p)]])
    return l2g

############### model utils ###############
def conv1d_bn(x, filters, name=None): # mlp for each IMU
    from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation
    from tensorflow.keras import regularizers
    x = Conv1D(filters, kernel_size=1, padding="valid",
              kernel_regularizer=regularizers.l2(0.01), name=name)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)

def dense_bn(x, filters):
    from tensorflow.keras.layers import Dense, BatchNormalization, Activation
    x = Dense(filters)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)

def bn_relu(x):
    from tensorflow.keras.layers import BatchNormalization, Activation
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)

def imu2vec(inputs, ks=16, dims=128, name=None):
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, Lambda, Reshape
    from tensorflow.keras import regularizers

    x = conv3d_bn(inputs, 64, kernel_size=[1,ks,3], strides=(1,1,3)) # (15,T,2,64)
    x = conv3d_bn(x, 64, kernel_size=[1,ks,2], strides=(1,1,1)) # (15,T,1,64)
    x = conv3d_bn(x, 64, kernel_size=[1,ks,1], strides=(1,1,1)) # (15,T,1,64)
    #x = Reshape(tuple(x.shape[1:3])+(-1,))(x) # (15,T,64)
    x = Lambda(lambda x: tf.squeeze(x, axis=-2))(x)

    dst = []
    gru = GRU(dims)
    for idx in range(inputs.shape[1]):
        dst.append(gru(x[:, idx]))
    return tf.stack(dst, axis=1)

def conv3d_bn(x, filters, kernel_size, strides, name=None):
    from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation
    from tensorflow.keras import regularizers
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding="valid",
              kernel_regularizer=regularizers.l2(0.0001), name=name)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)

def add_acc_noise(x, stddev):
    from tensorflow.keras.layers import GaussianNoise, Concatenate, Lambda
    #acceleration = Lambda(lambda x: x[:,:,:,:3,:])(x) # (batch, nb_imus, 120, 6, 1)
    #gyro = Lambda(lambda x: x[:,:,:,3:,:])(x)
    noise_acc = GaussianNoise(stddev)(x[:,:,:,:3])
    return Concatenate(axis=3)([noise_acc, x[:,:,:,3:]])