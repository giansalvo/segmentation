# https://www.kaggle.com/code/vidushibhatia/2-ultrasound-nerve-seg-unet-from-scratch


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os   # for data load
import datetime
import sys

# for reading and processing images
import imageio.v2 as imageio
from PIL import Image
import tifffile
# !pip install imagecodecs
import imagecodecs
import cv2

# for visualizations
import matplotlib.pyplot as plt

import numpy as np # for using np arrays
from numpy import asarray

# for bulding and running deep learning model
import tensorflow as tf
#from main import EPOCHS
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

from unet2 import create_model_UNet2

path1 = './ultrasound-nerve-segmentation_orig/train/'
path2 = './ultrasound-nerve-segmentation_orig/test/'
check = False
N_EPOCHS = 20
N_BATCH_SIZE = 64

tf.config.run_functions_eagerly(True) # giansalvo for dice_multiclass

def dice_multiclass(y_true, y_pred, epsilon=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X| + |Y|)
    """
    N_CLASSES = 3
    y_pred = tf.argmax(y_pred, -1)
    y_pred = tf.cast(y_pred, tf.uint8)
    y_pred = tf.squeeze(y_pred)

    # compute intermediate tensor for ground truth
    y_true = tf.cast(y_true, tf.uint8)
    y_true = tf.squeeze(y_true)

    dice = 0
    for i in range(N_CLASSES):
        pred_i = tf.cast(tf.equal(y_pred, i), tf.uint32) # subset of prediction for i class
        true_i = tf.cast(tf.equal(y_true, i), tf.uint32) # subset of mask for i class
        # temp = tf.equal(X, Y) # intersection of X and Y (boolean values)
        # temp = tf.cast(temp, tf.uint32)        # convert to 0/1
        inters_i = tf.keras.backend.eval(tf.reduce_sum(pred_i*true_i))
        union_i = tf.keras.backend.eval(tf.reduce_sum(pred_i)+tf.reduce_sum(true_i))

        #tf.print("\n"+str(i)+" " +str(inters_i)+" "+str(union_i))
        dice += (2. * inters_i + epsilon) / (union_i + epsilon)
    dice = dice / N_CLASSES

    # tf.print(str(pred_i*true_i))
    return dice

def LoadData (path1):
    # Read the images folder like a list
    image_dataset = os.listdir(path1)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    image_dataset.sort()
    for file in image_dataset:
        if file.endswith('_mask.tif'):
            mask_img.append(file)
            orig_img.append(file.replace("_mask.tif",".tif"))       

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    # orig_img.sort()
    # mask_img.sort()
    
    return orig_img, mask_img


print("Program start")
training_start = datetime.datetime.now().replace(microsecond=0)

img, mask = LoadData (path1)

show_images = 134

img_view  = imageio.imread(path1 + img[show_images])
mask_view = imageio.imread(path1 + mask[show_images])

print(img_view.shape)
print(mask_view.shape)
fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(img_view)
arr[0].set_title('Image ' + img[show_images])
arr[1].imshow(mask_view)
arr[1].set_title('Masked Image '+ mask[show_images])
if check:
    plt.show()
else:
    plt.close()

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
    y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    
    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h,i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        X[index] = single_img
        
        # convert mask into an array of desired shape (1 channel)
        
        single_mask_ind = mask[index]
        path = os.path.join(path1, single_mask_ind)
        single_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        single_mask = cv2.resize(single_mask, dsize=(m_h, m_w), interpolation=cv2.INTER_NEAREST)
        single_mask = asarray(single_mask)
        single_mask = single_mask[..., tf.newaxis]
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
        single_mask = single_mask/255
        single_mask = single_mask.astype(int) 
        y[index] = single_mask
    return X, y


# Define the desired shape
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]

# Process data using apt helper function
print ("Preprocess data...")
X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path1)

# QC the shape of output and classes in output dataset 
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 2 classes
print(np.unique(y))

# Visualize the output
image_index = 0
fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(X[image_index])
arr[0].set_title('Processed Image')
arr[1].imshow(y[image_index,:,:,0])
arr[1].set_title('Processed Masked Image ')
if check:
    plt.show()
else:
    plt.close()

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection


def UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=2):
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

print ("Split the dataset...")
# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

print ("Create and compile the network model...")
# Call the helper function for defining the layers for the model, given the input image size
unet = UNetCompiled(input_size=(128,128,3), n_filters=32, n_classes=3)
# unet = create_model_UNet2(output_channels=3, input_size=128, classes=2)

# Check the summary to better interpret how the output dimensions change in each layer
# unet.summary()


# There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
# Ideally, try different options to get the best accuracy
unet.compile(optimizer=tf.keras.optimizers.Adam(), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['sparse_categorical_accuracy', dice_multiclass])

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs_nerves", histogram_freq=1)
callbacks = [
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint("unet_nerves_orig.h5",
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)
]


# Run the model in a mini-batch fashion and compute the progress for each epoch
results = unet.fit(X_train, 
                    y_train, 
                    batch_size = N_BATCH_SIZE, 
                    epochs = N_EPOCHS, 
                    validation_data = (X_valid, y_valid),
                    callbacks = callbacks)
training_end = datetime.datetime.now().replace(microsecond=0)

# Save performances to file
fn_perf = "perf_nerves_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
print("Saving performances to file..." + fn_perf)
# Save invocation command line
print("Invocation command: ", end="", file=open(fn_perf, 'a'))
narg = len(sys.argv)
for x in range(narg):
    print(sys.argv[x], end = " ", file=open(fn_perf, 'a'))
print("\n", file=open(fn_perf, 'a'))
# Save performance information        
training_time = training_end - training_start
print("Training time: {}\n".format(training_time), file=open(fn_perf, 'a'))
model_history = results
for key in model_history.history.keys():
    print("{}: {:.4f}".format(key,  model_history.history[key][-1]), file=open(fn_perf, 'a'))

# Plot loss functions
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
fn_plot = "plot_nerves_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
print("Saving training plot to file..." + fn_plot)
plt.savefig(fn_plot)
if check:
    plt.show()
else:
    plt.close()

i = 0
sample_image = X[i]
ground_truth = y[i]
img = sample_image[np.newaxis, ...]
pred_y = unet.predict(img)
pred_mask = tf.argmax(pred_y[0], axis=-1)

fig, arr = plt.subplots(1, 3, figsize=(15, 15))
arr[0].imshow(sample_image)
arr[0].set_title('Image')
arr[1].imshow(ground_truth)
arr[1].set_title('Ground Truth')
arr[2].imshow(pred_mask)
arr[2].set_title('Prediction')

check = True
if check:
    plt.show()
else:
    plt.close()

# ## predict test set
# image_dataset = os.listdir(path2)
# test_img = []
# for file in image_dataset:
#     test_img.append(file)


# # create a table with X and Y
# m = len(test_img)                     # number of images
# i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image

# # Define X and Y as number of images along with shape of one image
# test_X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)

# # Resize images and masks
# for file in test_img:
#     # convert image into an array of desired shape (3 channels)
#     index = test_img.index(file)
#     path = os.path.join(path2, file)
#     single_img = Image.open(path).convert('RGB')
#     single_img = single_img.resize((i_h,i_w))
#     single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
#     single_img = single_img/256.
#     test_X[index] = single_img
    
# # predict masks
# test_y = []
# for img in test_X:
#     img = img[np.newaxis, ...]
#     pred_y = unet.predict(img)
#     pred_mask = tf.argmax(pred_y[0], axis=-1)
#     test_y.append(pred_mask)


# def rle_encoding(x):
#     '''
#     x: numpy array of shape (height, width), 1 - mask, 0 - background
#     Returns run length as list
#     '''
#     dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b>prev+1): run_lengths.extend((b+1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths


# import pandas as pd
# test_output = pd.DataFrame(columns = ['img','pixels'])

# for i,item in enumerate(test_y):
#     encoding = rle_encoding(item.numpy())
#     pixels = ' '.concat(map(str, encoding))
#     df = {'img': test_img[i][:-4], 'pixels': pixels}
#     test_output = test_output.append(df, ignore_index = True)
# test_output
# test_output.to_csv('./submission.csv', index=False)