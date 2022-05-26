"""
    Neural Network implementation for image segmentation

    For more information about autotune:
    https://www.tensorflow.org/guide/data_performance#prefetching

    Copyright (c) 2022 Giansalvo Gusinu <profgusinu@gmail.com>
    Copyright (c) 2021 Emil Zakirov and others
    Copyright (c) 2020 Yann LE GUILLY

    Code adapted from following articles/repositories:
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
import argparse
import datetime
import random
import sys
import os
import shutil
import logging
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image

from deeplab_v3_plus import create_model_deeplabv3plus
from unet import create_model_UNet
from unet2 import create_model_UNet2
from unet3 import create_model_UNet3

# CONSTANTS
ACTION_SPLIT = "split"
ACTION_TRAIN = "train"
ACTION_PREDICT = "predict"
ACTION_SUMMARY = "summary"
ACTION_EVALUATE = "evaluate"
ACTION_INSPECT = "inspect"
PNG_EXT = ".png"
FEXT_JPEG = "*.jpg"
MODEL_UNET = "unet"
MODEL_UNET2 = "unet2"
MODEL_UNET3 = "unet3"
MODEL_DEEPLABV3PLUS = "deeplabv3plus"
WEIGHTS_FNAME_DEFAULT = 'weights.h5'
REGEXP_DEFAULT = "*.png"
TRANSF_LEARN_IMAGENET_AND_FREEZE_DOWN = "imagenet_freeze_down"  # must match with unet2.py

# folders' structure
DATASET_IMG_SUBDIR = "images"
DATASET_ANNOT_SUBDIR = "annotations"
DATASET_TRAIN_SUBDIR = "training"
DATASET_VAL_SUBDIR = "validation"
DATASET_TEST_SUBDIR = "test"
DEFAULT_NETSTRUCTURE_PATH = 'model_saved/model_name'
DEFAULT_LOGS_DIR = "logs"

# global variables: default values
# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 1000
EPOCHS = 100
SEED = 42       # this allows to generate the same random numbers
IMG_SIZE = 128  # Image size that we are going to use
N_CHANNELS = 3  # Our images are RGB (3 channels)
N_CLASSES = 3   # Scene Parsing has 150 classes + `not labeled` (151)
BATCH_SIZE = 64
dataset_root_dir = ""
check = False
save_some_predictions = False
weights_fname = WEIGHTS_FNAME_DEFAULT
net_model = MODEL_UNET
network_structure_path = DEFAULT_NETSTRUCTURE_PATH
logs_dir = DEFAULT_LOGS_DIR
sample_image = None
sample_mask = None
model = None
learn_rate = 0.001
transfer_learning = None

# COPYRIGHT NOTICE AND PROGRAM VERSION
COPYRIGHT_NOTICE = "Copyright (C) 2022 Giansalvo Gusinu <profgusinu@gmail.com>"
PROGRAM_VERSION = "1.0"


# For each images of our dataset, we will apply some operations wrapped into
# a function. Then we will map the whole dataset with this function.
def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    print("Image path: " + img_path)
    mask_path = tf.strings.regex_replace(img_path, DATASET_IMG_SUBDIR, DATASET_ANNOT_SUBDIR)
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")   # TODO HARDCODED
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}


# Here we are using the decorator @tf.function
# if you want to know more about it:
# https://www.tensorflow.org/api_docs/python/tf/function
@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
  
  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels


@tf.function
def load_image(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


# Dice coeff (F1 score)
# https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_coef(y_true, y_pred, epsilon=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X| + |Y|)
         =  2*sum(|A*B|) / (sum(A^2) + sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum = K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1)
    coeff = (2. * intersection + epsilon) / (sum + epsilon)
    return 1 - coeff



# # Compute IoU coefficient (Jaccard index)
# def iou_coef(y_true, y_pred, smooth=1e-7):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

# Generalized dice loss for multi-class 3D segmentation
# # https://github.com/keras-team/keras/issues/9395
# def dice_coef_gattia(y_true, y_pred, smooth=1e-7):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# # https://github.com/keras-team/keras/issues/9395
# def dice_coef_multilabel_gattia(y_true, y_pred, numLabels=3):
#     print("ytrue.shape=" + str(y_true.shape))
#     print("ypred.shape=" + str(y_pred.shape))
#     dice=0
#     for index in range(numLabels):
#         print("index="+str(index))
#         dice -= dice_coef_gattia(y_true[index,:,:,:], y_pred[index,:,:,:])
#     return dice

# # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# # -> the score is computed for each class separately and then summed
# # alpha=beta=0.5 : dice coefficient
# # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# # alpha+beta=1   : produces set of F*-scores
# # implemented by E. Moebel, 06/04/18
# # https://github.com/keras-team/keras/issues/9395
# def tversky_loss(y_true, y_pred):
#     print("ytrue.shape=" + str(y_true.shape))
#     print("ypred.shape=" + str(y_pred.shape))
#     alpha = 0.5
#     beta  = 0.5
#     ones = K.ones(K.shape(y_true))
#     p0 = y_pred      # proba that voxels are class i
#     p1 = ones-y_pred # proba that voxels are not class i
#     g0 = y_true
#     g1 = ones-y_true
#     num = K.sum(p0*g0, (0,1,2))
#     den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
#     T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
#     Ncl = K.cast(K.shape(y_true)[-1], 'float32')
#     return Ncl-T

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_predictions(dataset=None, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if dataset:
        for image, true_mask in dataset.take(num):
            inference = model.predict(image)
            predictions = create_mask(inference)
            plot_samples_matplotlib([image[0], true_mask[0], predictions[0]])
    else:
        #plot_samples_matplotlib([sample_image[0], sample_mask[0]])
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
        inference = model.predict(one_img_batch)
        # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
        pred_mask = create_mask(inference)
        # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
        plot_samples_matplotlib([sample_image[0], sample_mask[0], pred_mask[0]])


def save_predictions(epoch, dataset=None, num=1):
    out_3map_fname = "figure_3map_{:03d}.jpg".format(epoch)
    out_seg_fname = "figure_seg_{:03d}.jpg".format(epoch)
    out_cmap_fname = "figure_cmap_{:03d}.png".format(epoch)

    prediction = infer(model=model, image_tensor=sample_image[0])
    # compute trimap output and save image to disk
    print("Saving trimap output segmented image to file: " + out_3map_fname)
    img1 = tf.cast(prediction, tf.uint8)
    img1 = tf.image.encode_jpeg(img1)
    tf.io.write_file(out_3map_fname, img1)

    # compute grayscale segmented image and save it to disk
    print("Saving grayscale segmented image to file: " + out_seg_fname)
    jpeg = generate_greyscale_image(prediction)
    tf.io.write_file(out_seg_fname, jpeg)

    # compute colored segmented image and save it to disk
    print("Saving colored segmented image to file: " + out_cmap_fname)
    im = generate_colormap_image(out_seg_fname)  # TODO this should take the 3map image in memory and not the greyscale file
    im.save(out_cmap_fname)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        if check:
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))
            show_predictions()
        if save_some_predictions:
            save_predictions(epoch=epoch)


def train_network(network_model, images_dataset, epochs, steps_per_epoch, validation_steps):

    logdir = os.path.join(logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    callbacks = [
        # to show samples after each epoch
        DisplayCallback(),
        # to collect some useful metrics and visualize them in tensorboard
        tensorboard_callback,
        # if no accuracy improvements we can stop the training directly
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        # to save checkpoints
        tf.keras.callbacks.ModelCheckpoint(weights_fname, 
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True)
    ]
    model_history = network_model.fit(images_dataset['train'], 
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps,
                                validation_data=images_dataset['val'],
                                callbacks=callbacks)
    return model_history


def read_image(image_path):
    img0 = tf.io.read_file(image_path)
    img0 = tf.image.decode_jpeg(img0, channels=3)
    img0 = tf.image.resize(img0, [IMG_SIZE,IMG_SIZE])
    return img0


def infer(model, image_tensor):
    img = np.expand_dims(image_tensor, axis=0)
    predictions = model.predict(img)
    predictions = create_mask(predictions)
    return predictions[0]


def generate_greyscale_image(img2):
    x_max = tf.reduce_max(img2)
    img2 = img2 * 255
    img2 = img2 / x_max
    jpeg = tf.image.encode_jpeg(tf.cast(img2, tf.uint8))
    return jpeg


def generate_colormap_image(input_fname):
    cm_hot = plt.cm.get_cmap('viridis')
    img_src = Image.open(input_fname).convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    return im

# def decode_segmentation_masks(mask, colormap, n_classes):
#     r = np.zeros_like(mask).astype(np.uint8)
#     g = np.zeros_like(mask).astype(np.uint8)
#     b = np.zeros_like(mask).astype(np.uint8)
#     for l in range(0, n_classes):
#         idx = mask == l
#         r[idx] = colormap[l, 0]
#         g[idx] = colormap[l, 1]
#         b[idx] = colormap[l, 2]
#     rgb = np.stack([r, g, b], axis=2)
#     return rgb


# def get_overlay(image, colored_mask):
#     image = tf.keras.preprocessing.image.array_to_img(image)
#     image = np.array(image).astype(np.uint8)
#     overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
#     return overlay


def plot_samples_matplotlib(display_list, labels_list=None, figsize=None):
    NIMG_PER_COLS = 6
    if figsize is None:
        PX = 1/plt.rcParams['figure.dpi']  # pixel in inches
        figsize = (600*PX, 300*PX)
    ntot = len(display_list)
    nrows = ntot // NIMG_PER_COLS + 1
    if ntot < NIMG_PER_COLS:
        ncols = ntot
    else:
        ncols = NIMG_PER_COLS
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i_img in range(ntot):
        i = i_img // NIMG_PER_COLS
        j = i_img % NIMG_PER_COLS
        if display_list[i_img].shape[-1] == 3:
                if nrows > 1:
                    if labels_list is not None:
                        axes[i, j].set_title(labels_list[i_img])
                    axes[i, j].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i_img]))
                else:
                    if labels_list is not None:
                        axes[i_img].set_title(labels_list[i_img])
                    axes[i_img].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i_img]))
        else:
                if nrows > 1:
                    axes[i, j].imshow(display_list[i_img])
                else:
                    axes[i_img].imshow(display_list[i_img])

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axes in axes.flat:
        axes.label_outer()

    plt.show()

#########################
# MAIN STARTS HERE
#########################
def main():
    global weights_fname
    global check
    global save_some_predictions
    global logs_dir
    global sample_image     #  used to display images during training
    global sample_mask      #  used to display images during training
    global model
    global learn_rate
    global transfer_learning

    # create logger
    logger = logging.getLogger('gians')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Starting")
    
    print("Program ver.: " + PROGRAM_VERSION)
    print(COPYRIGHT_NOTICE)
    executable = os.path.realpath(sys.executable)
    logger.info("Python ver.: " + sys.version)
    logger.info("Python executable: " + str(executable))
    logger.info("Tensorflow ver. " + str(tf.__version__))

    parser = argparse.ArgumentParser(
        description=COPYRIGHT_NOTICE,
        epilog = "Examples:\n"
                "       Prepare the dataset directories hierarchy starting from images/annotations initial directories:\n"
                "         $python %(prog)s split -ir initial_root_dir -dr dataset_root_dir\n"
                "         $python %(prog)s split -ir initial_root_dir -dr dataset_root_dir -s 0.4 0.3 0.3\n"
                "\n"
                "       Print the summary of the network model and save the model to disk:\n"
                "         $python %(prog)s summary -m deeplabv3plus\n"
                "\n"
                "       Train the network and write the weigths to disk:\n"
                "         $python %(prog)s train -m deeplabv3plus -dr dataset_dir \n"
                "         $python %(prog)s train -m deeplabv3plus -dr dataset_dir -w weigths_file.h5 --check\n"
                "\n"
                "       Make the network predict the segmented image of a given input image:\n"
                "         $python %(prog)s predict -m deeplabv3plus -i image.jpg -w weigths_file.h5 --check\n"
                "\n"
                "       Evaluate the network loss/accuracy performances based on the test set in the dataset directories hierarchy:\n"
                "         $python %(prog)s evaluate -m deeplabv3plus -dr dataset_dir -w weigths_file.h5 --check\n"
                "\n"
                "       Inspect some predictions from previous trainings:\n"
                "         $python %(prog)s inspect -r *.jpg\n"
                "\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-v", "--verbose", action="store_true")
    #group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("action", help="The action to perform: " +
                       ACTION_SPLIT+", "+ACTION_TRAIN+", "+ACTION_PREDICT+", "+ACTION_SUMMARY+", "+ACTION_EVALUATE,
                        choices=(ACTION_SPLIT, ACTION_TRAIN, ACTION_PREDICT, ACTION_SUMMARY, ACTION_EVALUATE, ACTION_INSPECT))
    parser.add_argument('--check', dest='check', default=False, action='store_true',
                        help="Display some images from dataset before training to check that dataset is ok.")
    parser.add_argument('--save_predictions', dest='save_some_predictions', default=False, action='store_true',
                        help="Save some prediction at the end of each epoch during training.")
    parser.add_argument('-ir', '--initial_root_dir', required=False, help="The initial root directory of images and trimaps.")
    parser.add_argument('-dr', '--dataset_root_dir', required=False, help="The root directory of the dataset.")
    parser.add_argument("-w", "--weigths_file", required=False, default=WEIGHTS_FNAME_DEFAULT,
                        help="The weigths file to be loaded/saved. It must be compatible with the network model chosen.")
    parser.add_argument("-i", "--input_image", required=False, help="The input file to be segmented.")
    parser.add_argument("-o", "--output_file", required=False, help="The output file with the segmented image.")
    parser.add_argument("-s", "--split_percentage", nargs=3, metavar=('train_p', 'validation_p', 'test_p' ),
                        type=float, default=[0.7, 0.2, 0.1],
                        help="The percentage of images to be copied respectively to train/validation/test set.")
    parser.add_argument("-e", "--epochs", required=False, default=EPOCHS, type=int, help="The number of times that the entire dataset is passed forward and backward through the network during the training")
    parser.add_argument("-b", "--batch_size", required=False, default=BATCH_SIZE, type=int, help="the number of samples that are passed to the network at once during the training")
    parser.add_argument('-m', "--model", required=False, default=MODEL_UNET, 
                        choices=(MODEL_UNET, MODEL_UNET2, MODEL_UNET3, MODEL_DEEPLABV3PLUS), 
                        help="The model of network to be created/used. It must be compatible with the weigths file.")
    parser.add_argument("-l", "--logs_dir", required=False, default=DEFAULT_LOGS_DIR, 
                        help="The directory where training information will be added. If it doesn't exist it will be created.")
    parser.add_argument("-nsp", "--network_structure_path", required=False, default=DEFAULT_NETSTRUCTURE_PATH, 
                        help="The path where the network structure will be saved (summary). If it doesn't exist it will be created.")
    parser.add_argument("-r", "--regexp", required=False, default=REGEXP_DEFAULT, 
                        help="Regular expression to be used to inspect.")
    parser.add_argument("-lr", "--learning_rate", required=False, type=float, default=learn_rate, 
                        help="The learning rate of the optimizer funtion during the training.")
    parser.add_argument('-tl', "--transfer_learning", required=False, default=None, 
                        choices=(TRANSF_LEARN_IMAGENET_AND_FREEZE_DOWN), 
                        help="The transfer learning option. Not all network models support all options.")
    parser.add_argument('-c', "--classes", required=False, default=N_CLASSES, type=int,
                        help="The number of possible classes that each pixel can belong to.")

    args = parser.parse_args()

    check = args.check
    save_some_predictions = args.save_some_predictions
    regexp = args.regexp
    weights_fname = args.weigths_file
    epochs = args.epochs
    batch_size = args.batch_size
    network_model = args.model
    network_structure_path = args.network_structure_path
    logs_dir = args.logs_dir
    learn_rate = args.learning_rate
    transfer_learning = args.transfer_learning
    classes_for_pixel = args.classes


    logger.debug("weights_fname=" + weights_fname)
    logger.debug("network_model=" + network_model)
    logger.debug("network_structure_path=" + network_structure_path)
    logger.debug("logs_dir=" + logs_dir)
    logger.debug("learn_rate=" + str(learn_rate))
    logger.debug("transfer_learning=" + str(transfer_learning))
    logger.debug("classes_for_pixel=" + str(classes_for_pixel))


    # create the unet network architecture with keras
    if network_model == MODEL_UNET:
        model = create_model_UNet(input_size=(IMG_SIZE, IMG_SIZE, N_CHANNELS), classes=classes_for_pixel, transfer_learning=transfer_learning)
    elif network_model == MODEL_UNET2:
        model = create_model_UNet2(output_channels=N_CHANNELS, input_size=IMG_SIZE, classes=classes_for_pixel, transfer_learning=transfer_learning)
    elif network_model == MODEL_UNET3:
        model = create_model_UNet3(input_shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS), classes=classes_for_pixel, transfer_learning=transfer_learning)
    elif network_model == MODEL_DEEPLABV3PLUS:
        model = create_model_deeplabv3plus(input_shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS), classes=classes_for_pixel, transfer_learning=transfer_learning)
    else:
        # BUG
        raise ValueError('BUG: Model of network not supported.')
     # optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)
    optimizer = Adam(learning_rate=learn_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy', dice_coef]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    if args.action == ACTION_TRAIN:
        if args.dataset_root_dir is None:
            raise ValueError('A value for dataset_root_dir paramenter (-dr) is required for training.')
        else:
            dataset_images_path =  os.path.join(args.dataset_root_dir, DATASET_IMG_SUBDIR)
        training_files_regexp =  os.path.join(dataset_images_path, DATASET_TRAIN_SUBDIR, FEXT_JPEG)
        validation_files_regexp = os.path.join(dataset_images_path, DATASET_VAL_SUBDIR, FEXT_JPEG)

        logger.debug("check=" + str(check))
        logger.debug("save_some_predictions=" + str(save_some_predictions))
        logger.debug("dataset_images_path=" + dataset_images_path)
        logger.debug("epochs=" + str(epochs))
        logger.debug("batch_size=" + str(batch_size))
        logger.debug("TRAINSET=" + training_files_regexp)

        # Creating a source dataset
        TRAINSET_SIZE = len(glob(training_files_regexp))
        print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

        VALSET_SIZE = len(glob(validation_files_regexp))
        print(f"The Validation Dataset contains {VALSET_SIZE} images.")

        if TRAINSET_SIZE == 0 or VALSET_SIZE == 0:
            print("ERROR: Training dataset and validation datasets must be not empty!")
            exit()

        STEPS_PER_EPOCH = TRAINSET_SIZE // batch_size
        VALIDATION_STEPS = VALSET_SIZE // batch_size

        logger.debug("STEPS_PER_EPOCH=" + str(STEPS_PER_EPOCH))
        logger.debug("VALIDATION_STEPS=" + str(VALIDATION_STEPS))
        if STEPS_PER_EPOCH == 0:
            print("ERROR: Not enough images for the training process!")
            exit()

        train_dataset = tf.data.Dataset.list_files(training_files_regexp, seed=SEED)
        train_dataset = train_dataset.map(parse_image)

        val_dataset = tf.data.Dataset.list_files(validation_files_regexp, seed=SEED)
        val_dataset = val_dataset.map(parse_image)

        dataset = {"train": train_dataset, "val": val_dataset}

        # -- Train Dataset --#
        dataset['train'] = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset['train'] = dataset['train'].cache()
        dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
        dataset['train'] = dataset['train'].batch(batch_size)
        dataset['train'] = dataset['train'].repeat()
        dataset['train'] = dataset['train'].map(Augment())
        dataset['train'] = dataset['train'].prefetch(buffer_size=tf.data.AUTOTUNE)

        # train_batches = (
        #     train_images
        #     .cache()
        #     .shuffle(BUFFER_SIZE)
        #     .batch(BATCH_SIZE)
        #     .repeat()
        #     .map(Augment())
        #     .prefetch(buffer_size=tf.data.AUTOTUNE))

        # -- Validation Dataset --#
        dataset['val'] = dataset['val'].map(load_image)
        dataset['val'] = dataset['val'].repeat()
        dataset['val'] = dataset['val'].batch(batch_size)
        dataset['val'] = dataset['val'].prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.debug(dataset['train'])
        logger.debug(dataset['val'])

        # how shuffle works: https://stackoverflow.com/a/53517848

        # Visualize the content of our dataloaders to make sure everything is fine.
        if check or save_some_predictions:
            print("Displaying content of dataset to make sure everything is fine...")
            for image, mask in dataset['train'].take(1):
                sample_image, sample_mask = image, mask
                plot_samples_matplotlib([sample_image[0], sample_mask[0]],
                                        ["Sample image", "Ground truth"])
        
        logger.info("Start network training...")
        model_history = train_network(model, dataset, epochs, STEPS_PER_EPOCH, VALIDATION_STEPS)
        logger.info("Network training end.")

        fn, fext = os.path.splitext(os.path.basename(weights_fname))
        # Save performances to file
        fn_perf = "perf_" + fn + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
        print("Saving performances to file..." + fn_perf)
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
        #  plt.ylim([0, 1])
        plt.legend()
        fn_plot = "plot_" + fn + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
        print("Saving plot to file..." + fn_plot)
        plt.savefig(fn)
        if check:
            plt.show()
            # show some predictions at the end of the training
            show_predictions(dataset['train'], 3)
        else:
            plt.close()


    elif args.action == ACTION_PREDICT:
        if args.input_image is None:
            print("ERROR: missing input_image parameter")
            exit()
        else:
            input_fname = args.input_image

        if args.output_file is None:
            output_fname = input_fname
        else:
            output_fname = args.output_file
        fn, fext = os.path.splitext(os.path.basename(output_fname))
        output_fname = fn + "_segm" + fext
        out_3map_fname = fn + "_3map" + fext
        out_cmap_fname = fn + "_cmap.png"
        logger.debug("output_fname=" + output_fname)

        model.load_weights(weights_fname)

        img0 = read_image(input_fname)
        
        prediction = infer(model=model, image_tensor=img0)
        
        # compute trimap output and save image to disk
        print("Saving trimap output segmented image to file: " + out_3map_fname)
        img1 = tf.cast(prediction, tf.uint8)
        img1 = tf.image.encode_jpeg(img1)
        tf.io.write_file(out_3map_fname, img1)

        # compute grayscale segmented image and save it to disk
        print("Saving grayscale segmented image to file: " + output_fname)
        jpeg = generate_greyscale_image(prediction)
        tf.io.write_file(output_fname, jpeg)
    
        # compute colored segmented image and save it to disk
        print("Saving colored segmented image to file: " + out_cmap_fname)
        im = generate_colormap_image(output_fname)  # TODO this should take the 3map image in memory and not the greyscale file
        im.save(out_cmap_fname)

        # if check is set, then display image to screen
        if check:
            plot_samples_matplotlib([img0, prediction])

    elif args.action == ACTION_SUMMARY:
        # print network's structure summary and save whole architecture plus weigths
        # import pydot
        # import graphviz
        tf.keras.utils.plot_model(model, to_file='model_summary.png', show_shapes=True, show_layer_names=True) #  TODO BUG graphical image doesn't get displayed
        model.summary()
        print("Model metrics names: " + str(model.metrics_names))
        model.save(network_structure_path)
    
    elif args.action == ACTION_SPLIT:
        # split images and trimap in train/validate/test by creating the dataset folder hierarchy
        train_p = args.split_percentage[0]
        val_p = args.split_percentage[1]
        test_p = args.split_percentage[2]
        if round(train_p + val_p + test_p) != 1:
            print(str(train_p + val_p + test_p))
            print(args.split_percentage)
            print("ERROR: the sum of percentages parameters must give 1")
            exit()
        if args.initial_root_dir is None:
            print("ERROR: you must specify the initial_root_dir parameter")
            exit()
        initial_root_dir = args.initial_root_dir
        if args.dataset_root_dir is None:
            print("ERROR: you must specify the dataset_root_dir parameter")
            exit()
        dataset_root_dir = args.dataset_root_dir
        if os.path.isdir(dataset_root_dir):
            print("ERROR: directory" + str(dataset_root_dir) + " already exists")
            exit()
        
        initial_images_dir = os.path.join(initial_root_dir, DATASET_IMG_SUBDIR)
        initial_annotations_dir = os.path.join(initial_root_dir, DATASET_ANNOT_SUBDIR)
        files_regexp = os.path.join(initial_images_dir, FEXT_JPEG)
        logger.debug(str(files_regexp))
  
        filenames = list(glob(files_regexp))
        ntot = len(filenames)
        ntrain = round(ntot * train_p)
        nvalid = round(ntot * val_p)
        ntest  = ntot - ntrain - nvalid  # avoid rounding errors
        print("Number of images found: ", ntot)
        if ntot == 0:
            print("ERROR: no images found with the following serch pattern")
            print(str(files_regexp))
            exit()
        
        print("Creating dataset directory structure...")
        os.mkdir(dataset_root_dir)
        # IMAGES SUBDIRS
        temp_root = os.path.join(dataset_root_dir, DATASET_IMG_SUBDIR)
        os.mkdir(temp_root)
        temp = os.path.join(temp_root, DATASET_TRAIN_SUBDIR)
        os.mkdir(temp)
        temp = os.path.join(temp_root, DATASET_VAL_SUBDIR)
        os.mkdir(temp)
        temp = os.path.join(temp_root, DATASET_TEST_SUBDIR)
        os.mkdir(temp)
        # ANNOTATIONS SUBDIRS
        temp_root = os.path.join(dataset_root_dir, DATASET_ANNOT_SUBDIR)
        os.mkdir(temp_root)
        temp = os.path.join(temp_root, DATASET_TRAIN_SUBDIR)
        os.mkdir(temp)
        temp = os.path.join(temp_root, DATASET_VAL_SUBDIR)
        os.mkdir(temp)
        temp = os.path.join(temp_root, DATASET_TEST_SUBDIR)
        os.mkdir(temp)
        print("Train set: " + str(ntrain) + 
            " - validation set: " + str(nvalid) +
            " - test set: " +str(ntest))
        logger.debug("first image: " + filenames[0])

        random.seed(SEED)
        indexes = random.sample(range(ntot), ntot)
        
        # TRAINING DATASET
        print("Copying images and trimaps of training set...")
        img_dir = os.path.join(dataset_root_dir, DATASET_IMG_SUBDIR, DATASET_TRAIN_SUBDIR)
        annot_dir = os.path.join(dataset_root_dir, DATASET_ANNOT_SUBDIR, DATASET_TRAIN_SUBDIR)
        for i in range(ntrain):
            j = indexes[i]
            shutil.copy2(filenames[j], img_dir)
            fn, _ = os.path.splitext(os.path.basename(filenames[j]))
            fn = os.path.join(initial_annotations_dir, fn + PNG_EXT)
            shutil.copy2(fn, annot_dir)

        # VALIDATION DATASET
        print("Copying images and trimaps of validation set...")
        img_dir = os.path.join(dataset_root_dir, DATASET_IMG_SUBDIR, DATASET_VAL_SUBDIR)
        annot_dir = os.path.join(dataset_root_dir, DATASET_ANNOT_SUBDIR, DATASET_VAL_SUBDIR)
        for i in range(nvalid):
            j = indexes[ntrain + i]
            shutil.copy2(filenames[j], img_dir)
            fn, _ = os.path.splitext(os.path.basename(filenames[j]))
            fn = os.path.join(initial_annotations_dir, fn + PNG_EXT)
            shutil.copy2(fn, annot_dir)

        # TEST DATASET
        print("Copying images and trimaps of test set...")
        img_dir = os.path.join(dataset_root_dir, DATASET_IMG_SUBDIR, DATASET_TEST_SUBDIR)
        annot_dir = os.path.join(dataset_root_dir, DATASET_ANNOT_SUBDIR, DATASET_TEST_SUBDIR)
        for i in range(ntest):
            j = indexes[ntrain + nvalid + i] 
            shutil.copy2(filenames[j], img_dir)
            fn, _ = os.path.splitext(os.path.basename(filenames[j]))
            fn = os.path.join(initial_annotations_dir, fn + PNG_EXT)
            shutil.copy2(fn, annot_dir)

    elif args.action == ACTION_EVALUATE:
        if args.dataset_root_dir is None:
            print("ERROR: you must specify the initial_root_dir parameter")
            exit()
        else:
            dataset_images_path =  os.path.join(args.dataset_root_dir, DATASET_IMG_SUBDIR)
        test_files_regexp =  os.path.join(dataset_images_path, DATASET_TEST_SUBDIR, FEXT_JPEG)

        logger.debug("check=" + str(check))
        logger.debug("dataset_images_path=" + dataset_images_path)
        logger.debug("batch_size=" + str(batch_size))
        logger.debug("testset=" + test_files_regexp)

        # Creating a source dataset
        testset_size = len(glob(test_files_regexp))
        print(f"The test Dataset contains {testset_size} images.")

        if testset_size == 0:
            print("ERROR: the test datasets must be not empty!")
            exit()

        steps_num = testset_size // batch_size
        logger.debug("steps_num=" + str(steps_num))
        if steps_num == 0:
            print("ERROR: steps_num cannot be zero. Increase number of images or reduce batch_size.")
            exit()

        test_dataset = tf.data.Dataset.list_files(test_files_regexp, seed=SEED)
        test_dataset = test_dataset.map(parse_image)

        # -- test Dataset --#
        test_dataset = test_dataset.map(load_image)
        test_dataset = test_dataset.repeat()
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        scores = model.evaluate(test_dataset,
                                steps = steps_num)
        print(str(dict(zip(model.metrics_names, scores))))
    elif args.action == ACTION_INSPECT:
        logger.debug("regexp=" + regexp)
        filenames_regexp = regexp
        num = len(glob(filenames_regexp))
        print(f"Found {num} images.")
        if num < 2:
            print("ERROR: You need at least two images.")
            exit()

        images = []
        fnames = []
        for fname in glob(filenames_regexp):
            img = tf.io.read_file(fname)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.uint8)
            images.append(img)
            fnames.append(fname)

        plot_samples_matplotlib(images, fnames)

    print("Program terminated correctly.")
    logger.debug("Program end.")

if __name__ == '__main__':
    main()
