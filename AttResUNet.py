# coding: utf-8
# code adapted from MoleImg https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

'''
Hyper-parameters
'''
# network structure
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 2 # size of upsampling filters

TRANSF_LEARN_FREEZE_ENCODER = "freeze_encoder"
TRANSF_LEARN_FREEZE_DECODER = "freeze_decoder"


# '''
# Definitions of loss and evaluation metrices
# '''

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


# def jacard_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


# def jacard_coef_loss(y_true, y_pred):
#     return -jacard_coef(y_true, y_pred)


# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


def expend_as(tensor, rep):
     return tf.keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False, trainable=True):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    axis = 3
    conv = tf.keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')
    conv.trainable = trainable
    conv = conv(x)
    if batch_norm is True:
        conv = tf.keras.layers.BatchNormalization(axis=axis)(conv)
    conv1 = tf.keras.layers.Activation('relu')
    conv1.trainable = trainable
    conv1=conv1(conv)
    conv2 = tf.keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')
    conv2.trainable = trainable
    conv2 = conv2(conv1)
    if batch_norm is True:
        conv2 = tf.keras.layers.BatchNormalization(axis=axis)(conv2)
    activ = tf.keras.layers.Activation('relu')
    activ.trainable = trainable
    activ=activ(conv2)
    if dropout > 0:
        activ = tf.keras.layers.Dropout(dropout)(activ)

    shortcut = tf.keras.layers.Conv2D(size, kernel_size=(1, 1), padding='same')
    shortcut.trainable = trainable
    shortcut = shortcut(x)
 
    if batch_norm is True:
        shortcut = tf.keras.layers.BatchNormalization(axis=axis)(shortcut)

    res_path = tf.keras.layers.add([shortcut, activ])
    return res_path

def gating_signal(input, out_size, batch_norm=False, trainable=True):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = tf.keras.layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = tf.keras.layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = tf.keras.layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = tf.keras.layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = tf.keras.layers.add([upsample_g, theta_x])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = tf.keras.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = tf.keras.layers.multiply([upsample_psi, x])

    result = tf.keras.layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = tf.keras.layers.BatchNormalization()(result)
    return result_bn

def create_model_AttentionResUNet(input_size=(128, 128, 3), n_filters=32, n_classes=2, transfer_learning=None):
    '''
    Residual UNet construction, with attention gate
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: model
    '''
    train_decoder = True
    train_encoder = True
    if transfer_learning == TRANSF_LEARN_FREEZE_ENCODER:
        print("AttResUNet.py: transfer_learning: freeze encoder")
        train_encoder = False
    elif transfer_learning == TRANSF_LEARN_FREEZE_DECODER:
        print("AttResUNet.py: transfer_learning: freeze decoder")
        train_decoder = False

    # input data
    # dimension of the image depth
    inputs = tf.keras.layers.Input(input_size, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, n_filters, dropout=0, batch_norm=True, trainable=train_decoder)
    pool_64 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*n_filters, dropout=0, batch_norm=True, trainable=train_decoder)
    pool_32 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*n_filters, dropout=0, batch_norm=True, trainable=train_decoder)
    pool_16 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*n_filters, dropout=0.3, batch_norm=True, trainable=train_decoder)
    pool_8 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*n_filters, dropout=0.3, batch_norm=True, trainable=train_decoder)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*n_filters, batch_norm=True, trainable=train_encoder)
    att_16 = attention_block(conv_16, gating_16, 8*n_filters)
    up_16 = tf.keras.layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = tf.keras.layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*n_filters, dropout=0, batch_norm=True, trainable=train_encoder)
    up_conv_16.trainable = transfer_learning != TRANSF_LEARN_FREEZE_ENCODER
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*n_filters, batch_norm=True, trainable=train_encoder)
    att_32 = attention_block(conv_32, gating_32, 4*n_filters)
    up_32 = tf.keras.layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = tf.keras.layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*n_filters, dropout=0, batch_norm=True, trainable=train_encoder)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*n_filters, batch_norm=True, trainable=train_encoder)
    att_64 = attention_block(conv_64, gating_64, 2*n_filters)
    up_64 = tf.keras.layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = tf.keras.layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*n_filters, dropout=0, batch_norm=True, trainable=train_encoder)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, n_filters, batch_norm=True, trainable=train_encoder)
    att_128 = attention_block(conv_128, gating_128, n_filters)
    up_128 = tf.keras.layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = tf.keras.layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, n_filters, dropout=0, batch_norm=True, trainable=train_encoder)

    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = tf.keras.layers.Conv2D(n_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = tf.keras.layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = tf.keras.layers.Activation('relu')(conv_final)

    # Model integration
    model = tf.keras.Model(inputs, conv_final, name="AttentionResUNet")
    return model
