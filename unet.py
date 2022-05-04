"""
    Neural Network implementation for image segmentation

    Copyright (c) 2022 Giansalvo Gusinu <profgusinu@gmail.com>
    Copyright (c) 2020 Yann LE GUILLY

    Code adapted from colab and article found here
    https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
    https://github.com/dhassault/tf-semantic-example

    Permission is hereby granted, free of charge, to any person obtaining a 
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
import tensorflow as tf
from tensorflow.keras.layers import *

def create_model_UNet(input_size=(128, 128, 3), classes=150):
    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    # Everything here is from tensorflow.keras.layers
    # I imported tensorflow.keras.layers * to make it easier to read
    dropout_rate = 0.5  # TODO REMOVE NOT USED???

    # If you want to know more about why we are using `he_normal`:
    # https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849
    # Or the excelent fastai course:
    # https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb
    initializer = 'he_normal'

    # -- Encoder -- #
    # Block encoder 1
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_1)
    # Block encoder 2
    max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_2)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)
    # Block  encoder 3
    max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
    # Block  encoder 4
    max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
    # -- End Encoder -- #
    # ----------- #
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
    conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(maxpool)
    conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv)
    # ----------- #
    # -- Decoder -- #
    # Block decoder 1
    up_dec_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv))
    merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis=3)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)
    # Block decoder 2
    up_dec_2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_1))
    merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis=3)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_2)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)
    # Block decoder 3
    up_dec_3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_2))
    merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis=3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)
    # Block decoder 4
    up_dec_4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_3))
    merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis=3)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_4)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    conv_dec_4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    # -- End Decoder -- #
    output = Conv2D(classes, 1, activation='softmax')(conv_dec_4)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="U-Net")
   
    return model