import sys
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Input,
                            Concatenate, GlobalMaxPooling1D, Lambda,
                            MultiHeadAttention, LayerNormalization)
from tensorflow.keras import Model
from utils import imu2vec, conv1d_bn, add_acc_noise
import myFlags
from absl import flags
FLAGS = flags.FLAGS

def build_pointnet_model(nb_classes, input_shape=(13, 120, 6, 1), params=None):
    inputs = Input(input_shape)
    x = add_acc_noise(inputs, 1.0) # add noise
    
    # IMU-wise feature extraction
    u = imu2vec(x, FLAGS.imu2vec_kernelsize)
    # pointnet
    u = pointnet(u, FLAGS.mlp_nodes)
    u = Conv1D(64, 1, name="penultimate")(u)
    preds = Conv1D(nb_classes, 1, activation="softmax", name="preds")(u[:,1:])

    model = Model(inputs=inputs, outputs=preds)
    opt = tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model

def build_transformer_model(nb_classes, input_shape=(13, 120, 6, 1), params=None):
    inputs = Input(input_shape)
    x = add_acc_noise(inputs, 1.0) # add noise
    
    # IMU-wise feature extraction
    u = imu2vec(x, FLAGS.imu2vec_kernelsize, 128)
    u = conv1d_bn(u, FLAGS.mlp_nodes)
    u = conv1d_bn(u, FLAGS.mlp_nodes)

    # transformer
    tf_input_dims = u.shape[-1]
    for i in range(FLAGS.nb_transformer_layers):
        u = transformer_encoder(u, i, FLAGS.embed_dims,
                FLAGS.ff_dims, tf_input_dims, rate=0.1)
    u = Conv1D(64, 1, name="penultimate")(u)
    preds = Conv1D(nb_classes, 1, activation="softmax", name="preds")(u[:,1:])

    model = Model(inputs=inputs, outputs=preds)
    opt = tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model

def build_pointnet_transformer_model(nb_classes, input_shape=(13, 120, 6, 1)):
    inputs = Input(input_shape)
    x = add_acc_noise(inputs, 1.0) # add noise
    
    # IMU-wise feature extraction
    u = imu2vec(x, FLAGS.imu2vec_kernelsize, 128)
    # pointnet
    u = pointnet(u, FLAGS.mlp_nodes)
    
    # transformer
    tf_input_dims = u.shape[-1]
    for i in range(FLAGS.nb_transformer_layers):
        u = transformer_encoder(u, i, FLAGS.embed_dims,
                FLAGS.ff_dims, tf_input_dims, rate=0.1)
    u = Conv1D(64, 1, name="penultimate")(u)
    preds = Conv1D(nb_classes, 1, activation="softmax", name="preds")(u[:,1:]) # except for root IMU

    model = Model(inputs=inputs, outputs=preds)
    opt = tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model

def pointnet(x, mlp_nodes, name=None):
    nb_imus = x.shape[1]
    lf = conv1d_bn(x, mlp_nodes)
    gf = GlobalMaxPooling1D()(lf)
    gf = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=1), [1,nb_imus,1]))(gf)
    gf = Concatenate(axis=2)([x, lf, gf])
    return conv1d_bn(gf, mlp_nodes)

def transformer_encoder(x, i, embed_dims, ff_dims, input_dims, rate=0.1):
    att = LayerNormalization(epsilon=1e-6)(x)
    att, _ = MultiHeadAttention(num_heads=4, key_dim=embed_dims,
                 name="attention{}".format(i))(att, att, return_attention_scores=True)
    out1 = Dropout(rate)(x + att)
    ffn_out = LayerNormalization(epsilon=1e-6)(out1)
    ffn_out = Dense(ff_dims, activation="relu")(ffn_out)
    ffn_out = Dense(input_dims)(ffn_out)
    out2 = Dropout(rate)(out1 + ffn_out)
    return out2