"""
    Neural Network implementation for image segmentation

    Copyright (c) 2022 Giansalvo Gusinu <profgusinu@gmail.com>
 
    Code adapted from TensorFlow Tutorial
 
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
from tensorflow_examples.models.pix2pix import pix2pix

TRANSF_LEARN_IMAGENET_AND_FREEZE_DOWN = "imagenet_freeze_down"

def create_model_UNet2(output_channels:int, input_size=128, classes=3, transfer_learning=None):
  print("unet2.py: WARNING parameter 'classes' not used. Reserved for future uses.")   # TODO parameter classes not used
  if transfer_learning == TRANSF_LEARN_IMAGENET_AND_FREEZE_DOWN:
    print("unet2.py: applying transfer learning: imagenet and freeze down stack.")
    w = "imagenet"
  else:
    w = None
  base_model = tf.keras.applications.MobileNetV2(input_shape=[input_size, input_size, 3],
                                                include_top=False,
                                                weights=w)

  # Use the activations of these layers
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

  if transfer_learning == TRANSF_LEARN_IMAGENET_AND_FREEZE_DOWN:
    down_stack.trainable = False

  up_stack = [
      pix2pix.upsample(512, 3),  # 4x4 -> 8x8
      pix2pix.upsample(256, 3),  # 8x8 -> 16x16
      pix2pix.upsample(128, 3),  # 16x16 -> 32x32
      pix2pix.upsample(64, 3),   # 32x32 -> 64x64
  ]

  inputs = tf.keras.layers.Input(shape=[input_size, input_size, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
