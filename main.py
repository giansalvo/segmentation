"""
    U-Net Neural Network implementation for image segmentation

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
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# CONSTANTS
ACTION_SPLIT = "split"
ACTION_TRAIN = "train"
ACTION_PREDICT = "predict"
ACTION_SUMMARY = "summary"
PNG_EXT = ".png"
FEXT_JPEG = "*.jpg"

# For more information about autotune:
# https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE

# DEFAULT PARAMETERS
WEIGHTS_FNAME_DEFAULT = 'unet_weights.h5'

# important for reproducibility
# this allows to generate the same random numbers
SEED = 42

# Image size that we are going to use
IMG_SIZE = 128
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 151

BATCH_SIZE = 32

# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 1000

EPOCHS = 20  # gians original it was 20

# gians PD folders structure
dataset_root_dir = "./dataset"
DATASET_IMG_SUBDIR = "images"
DATASET_ANNOT_SUBDIR = "annotations"
DATASET_TRAIN_SUBDIR = "training"
DATASET_VAL_SUBDIR = "validation"
DATASET_TEST_SUBDIR = "test"
PATH_SAVED_MODEL = 'model_saved/unet_model'
LOGS_DIR = "logs"

# global variables
check = False
weights_fname = WEIGHTS_FNAME_DEFAULT

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
    return input_image, input_mask


@tf.function
def load_image_train(datapoint: dict) -> tuple:
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

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

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


def display_sample(display_list, title_list=['Input Image', 'True Mask', 'Predicted Mask']):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title_list[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


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
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], true_mask, create_mask(pred_mask)])
    else:
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
        display_sample([sample_image[0], sample_mask[0],
                        pred_mask[0]])


def create_model_UNet():
    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    # Everything here is from tensorflow.keras.layers
    # I imported tensorflow.keras.layers * to make it easier to read
    dropout_rate = 0.5  # TODO REMOVE NOT USED???
    input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

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
    output = Conv2D(N_CLASSES, 1, activation='softmax')(conv_dec_4)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="U-Net")
    # optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)
    optimizer = Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        if check:
            show_predictions()
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


def train_network(network_model, images_dataset, steps_per_epoch, validation_steps):
    logdir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
                                epochs=EPOCHS,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps,
                                validation_data=images_dataset['val'],
                                callbacks=callbacks)


#########################
# MAIN STARTS HERE
#########################
def main():
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
                "         $python %(prog)s split -ir initial_root_dir -dr dataset_root_dir\n"
                "         $python %(prog)s split -ir initial_root_dir -dr dataset_root_dir -s 0.4 0.3 0.3\n"
                "\n"
                "         $python %(prog)s summary\n"
                "\n"
                "         $python %(prog)s train -dr dataset_dir\n"
                "         $python %(prog)s train -dr dataset_dir -w weigths_file.h5 --check\n"
                "\n"
                "         $python %(prog)s predict -i image.jpg -w weigths_file.h5 -o image_segm.png\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    group = parser.add_mutually_exclusive_group()
    #group.add_argument("-v", "--verbose", action="store_true")
    #group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("action", help="The action to perform: "
                                       + ACTION_SPLIT + ", " + ACTION_TRAIN + ", " + ACTION_PREDICT + ", " + ACTION_SUMMARY,
                        choices=(ACTION_SPLIT, ACTION_TRAIN, ACTION_PREDICT, ACTION_SUMMARY))
    parser.add_argument('--check', dest='check', default=False, action='store_true',
                        help="Display some images from dataset before training to check that dataset is ok")
    parser.add_argument('-ir', '--initial_root_dir', required=False, help="The initial root directory of images and trimaps")
    parser.add_argument('-dr', '--dataset_root_dir', required=False, help="The root directory of the dataset")
    parser.add_argument("-w", "--weigths_file", required=False, default=WEIGHTS_FNAME_DEFAULT,
                        help="The weigths file to be loaded/saved")
    parser.add_argument("-i", "--input_image", required=False, help="The input file to be segmented")
    parser.add_argument("-o", "--output_file", required=False, help="The output file with the segmented image")
    parser.add_argument("-s", "--split_percentage", nargs=3, metavar=('train_p', 'validation_p', 'test_p' ),
                        type=float, default=[0.4, 0.3, 0.3],
                        help="The percentage of images to be copied respectively to train/validation/test set")

    args = parser.parse_args()

    check = args.check
    weights_fname = args.weigths_file
    
    logger.debug("weights_fname=" + weights_fname)
    
    # create the unet network architecture with keras
    model = create_model_UNet()
    
    if args.action == ACTION_TRAIN:
        if args.dataset_root_dir is None:
            dataset_images_path = os.path.join(dataset_root_dir, DATASET_IMG_SUBDIR)
        else:
            dataset_images_path =  os.path.join(args.dataset_root_dir, DATASET_IMG_SUBDIR)
        training_files_regexp =  os.path.join(dataset_images_path, DATASET_TRAIN_SUBDIR, FEXT_JPEG)
        validation_files_regexp = os.path.join(dataset_images_path, DATASET_VAL_SUBDIR, FEXT_JPEG)

        logger.debug("check=" + str(check))
        logger.debug("dataset_images_path=" + dataset_images_path)

        logger.debug("TRAINSET=" + training_files_regexp)
        # Creating a source dataset
        TRAINSET_SIZE = len(glob(training_files_regexp))
        print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

        VALSET_SIZE = len(glob(validation_files_regexp))
        print(f"The Validation Dataset contains {VALSET_SIZE} images.")

        if TRAINSET_SIZE == 0 or VALSET_SIZE == 0:
            print("ERROR: Training dataset and validation datasets must be not empty!")
            exit()

        STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
        VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

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
        dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
        dataset['train'] = dataset['train'].repeat()
        dataset['train'] = dataset['train'].batch(BATCH_SIZE)
        dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

        # -- Validation Dataset --#
        dataset['val'] = dataset['val'].map(load_image_test)
        dataset['val'] = dataset['val'].repeat()
        dataset['val'] = dataset['val'].batch(BATCH_SIZE)
        dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

        logger.debug(dataset['train'])
        logger.debug(dataset['val'])

        # how shuffle works: https://stackoverflow.com/a/53517848

        # Visualize the content of our dataloaders to make sure everything is fine.
        if check:
            print("Displaying content of dataset to make sure everything is fine and exit...")
            for image, mask in dataset['train'].take(1):
                sample_image, sample_mask = image, mask
            display_sample([sample_image[0], sample_mask[0]])
            exit()

  
        train_network(model, dataset, STEPS_PER_EPOCH, VALIDATION_STEPS)
    elif args.action == ACTION_PREDICT:
        if args.input_image is None:
            print("ERROR: missing input_image parameter")
            exit()
        else:
            input_fname = args.input_image

        if args.output_file is None:
            fn, fext = os.path.splitext(os.path.basename(input_fname))
            output_fname = fn + "_segm" + fext
            out_3map_fname = fn + "_3map" + fext
        else:
            output_fname = args.output_file
            fn, fext = os.path.splitext(os.path.basename(output_fname))
            out_3map_fname = fn + "_3map" + fext
        logger.debug("output_fname=" + output_fname)

        model.load_weights(weights_fname)
        img0 = tf.io.read_file(input_fname)
        img0 = tf.image.decode_jpeg(img0, channels=3)
        img0 = tf.image.resize(img0, [IMG_SIZE,IMG_SIZE])
        # plt.imshow(img0)
        img = np.expand_dims(img0, axis=0)
        inference = model.predict(img)
        pred_mask = create_mask(inference)
        # compute trimap output and save image to disk
        img1 = pred_mask[0]
        print("Saving trimap output segmented image to file: " + out_3map_fname)
        img1 = tf.cast(img1, tf.uint8)
        img1 = tf.image.encode_jpeg(img1)
        tf.io.write_file(out_3map_fname, img1)
        # compute visible output and save image to disk
        img2 = pred_mask[0]
        print("Saving visible output segmented image to file: " + output_fname)
        x_max = tf.reduce_max(img2)
        img2 = img2 * 255
        img2 = img2 / x_max
        img2 = tf.cast(img2, tf.uint8)
        img2 = tf.image.encode_jpeg(img2)
        tf.io.write_file(output_fname, img2)

        # if check is set, then display image to screen
        if check:
            display_sample([img0, pred_mask[0]],
                           ['Input Image', 'Predicted Mask'])
            plt.show()

    elif args.action == ACTION_SUMMARY:
        # print network's structure summary and save hole architecture plus weigths 
        model.summary()
        model.save(PATH_SAVED_MODEL)
    
    elif args.action == ACTION_SPLIT:
        # split images and trimap in train/validate/test by creating the dataset folder hierarchy
        train_p = args.split_percentage[0]
        val_p = args.split_percentage[1]
        test_p = args.split_percentage[2]
        if train_p + val_p + test_p != 1:
            print("ERROR: the sum of percentages parameters must give 1")
            print(args.split_percentage)
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
        ntest  = round(ntot * test_p)
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
        # for i in range(10):
        #     j = indexes[i]
        #     print(str(j) + " - " + filenames[j])

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

    print("Program terminated correctly.")


if __name__ == '__main__':
    main()
