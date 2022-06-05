"""
Copyright (C) 2022 Giansalvo Gusinu <profgusinu@gmail.com>

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
import logging
import os
import time
from venv import logger
import cv2
import numpy as np

import tensorflow as tf

# COPYRIGHT NOTICE AND PROGRAM VERSION
COPYRIGHT_NOTICE = "Copyright (C) 2022 Giansalvo Gusinu"
PROGRAM_VERSION = "1.0"

# CONSTANTS
ACTION_ANONYMIZE = "anonymize"
ACTION_CROP = "crop"
ACTION_MASK = "mask"
ACTION_MEASURE = "measure"
ACTION_TRIMAP = "trimap"
CROP_X_DEFAULT = 330
CROP_Y_DEFAULT = 165
CROP_W_DEFAULT = 300
CROP_H_DEFAULT = 300
ANONYMIZE_X_DEFAULT = 0
ANONYMIZE_Y_DEFAULT = 0
ANONYMIZE_W_DEFAULT = 1240
ANONYMIZE_H_DEFAULT = 100

# FOLDERS
IMAGES_ORIG = "images_jpg"
IMAGES_SUBDIR = "images"
ANNOTATIONS_ORIG = "annotations_jpg"
ANNOTATIONS_SUBDIR = "annotations"

# VALUES
VALUE_FOREGROUND    = 1
VALUE_BACKGROUND    = 3

# COLOUR MASKS
cyan_lower = np.array([34, 85, 30])
cyan_upper = np.array([180, 252, 234])
white_lower = np.array([0, 0, 255])
white_upper = np.array([180, 255, 255])
green_lower = np.array([1, 0, 0])
green_upper = np.array([80, 255, 255])
contour_color = (0, 255, 0)  # green contour (BGR)
fill_color = list(contour_color)

# DIRECTORIES
SUBDIR_WHITE = "white"
SUBDIR_CYAN = "cyan"
SUBDIR_BINARY = "bin"
SUBDIR_VISIBLE = "visible"

def measure_area(image_rgb, color_rgb):
    # Find all pixels where the 3 RGB values match "color", and count them
    result = np.count_nonzero(np.all(image_rgb == color_rgb, axis=2))
    return result


def anonimize(image, x=ANONYMIZE_X_DEFAULT, y=ANONYMIZE_Y_DEFAULT, 
                w=ANONYMIZE_W_DEFAULT, h=ANONYMIZE_H_DEFAULT):
    # Draw black background rectangle in the upper region of the image
    # _, w, _ = image.shape
    # x, y, w, h = 0, 0, w, 40
    cv2.rectangle(image, (x, x), (x + w, y + h), (0, 0, 0), -1)
    return image


def put_text(image, text):
    # blue = (209, 80, 0, 255),  # font color
    white = (255, 255, 255, 255)  # font color
    x, y, w, h = 10, 40, 20, 40
    # Draw black background rectangle
    cv2.rectangle(image, (x, x), (x + w, y + h), (0, 0, 0), -1)
    cv2.putText(
        image,  # numpy array on which text is written
        text,  # text
        (x, y),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        1,  # font size
        white,  # font color
        1)  # font stroke
    return image


def fill_contours_white(finput, num_extra_iteration=4):
    lower_color = white_lower
    upper_color = white_upper

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2
    #fn, fext = os.path.splitext(os.path.basename(finput))

    img_orig = cv2.imread(finput)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # PASS 1: Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS 1: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    # cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS 1: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    for x in range(num_extra_iteration):
        # Close contour
        imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(imghsv, green_lower, green_upper)
        # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # iteration=2 NOT GOOD!
        img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find outer contours
        cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
        cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)
        img_green_seg = img_contours

    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)
    return img_filled


def fill_contours_cyan(finput, num_extra_iteration=3):
    lower_color = cyan_lower
    upper_color = cyan_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2

    img_orig = cv2.imread(finput)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # PASS 1: Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS 1: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS 1: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    for x in range(num_extra_iteration):
        # Close contour
        imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(imghsv, green_lower, green_upper)
        # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # iteration=2 NOT GOOD!
        img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find outer contours
        cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
        cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)
        img_green_seg = img_contours
    
    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)
    return img_filled

def fill_contours_all_files(input_directory, output_directory):
    ext = ('.jpg', '.jpeg', '.png')
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))

            # white contours
            img_filled = fill_contours_white(input_directory + "/" + fname)
            # Save the file
            subdir = output_directory + "/" + SUBDIR_WHITE + "/"
            cv2.imwrite(subdir + fn + ".png", img_filled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cyan contours
            img_filled = fill_contours_cyan(input_directory + "/" + fname)
            # Save the file
            subdir = output_directory + "/" + SUBDIR_CYAN + "/"
            cv2.imwrite(subdir + fn + ".png", img_filled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return


def generate_trimaps_all_files(input_directory, output_directory):
    ext = ('.jpg', '.jpeg', '.png')
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))
            img_visible, img_binary = generate_trimap(input_directory + "/" + fname)
            subdir = output_directory + "/" + SUBDIR_VISIBLE + "/"
            cv2.imwrite(subdir + fn + ".png", img_visible, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            subdir = output_directory + "/" + SUBDIR_BINARY + "/"
            cv2.imwrite(subdir + fn + ".png", img_binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return


def generate_trimap(fname, erosion_iter=6, dilate_iter=6):
    mask = cv2.imread(fname, 0)
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 128
    # threshold the image
    mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
    mask[mask == 1] = 255
    d_kernel = np.ones((3, 3))
    erode = cv2.erode(mask, d_kernel, iterations=erosion_iter)
    dilate = cv2.dilate(mask, d_kernel, iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode, mask)
    unknown2 = cv2.bitwise_xor(dilate, mask)
    unknowns = cv2.add(unknown1, unknown2)
    unknowns[unknowns == 255] = 127
    trimap = cv2.add(mask, unknowns)
    # cv2.imwrite("mask.png",mask)
    # cv2.imwrite("dilate.png",dilate)
    # cv2.imwrite("tri.png",trimap)
    labels = trimap.copy()
    labels[trimap == 127] = 1  # unknown
    labels[trimap == 255] = 2  # foreground
    return trimap, labels


def anonymize_all_files(input_directory, output_directory, 
                x=ANONYMIZE_X_DEFAULT, y=ANONYMIZE_X_DEFAULT, 
                w=ANONYMIZE_W_DEFAULT, h=ANONYMIZE_H_DEFAULT):
    ext = ('.jpg', '.jpeg', '.png')
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))
            img = cv2.imread(input_directory + "/" + fname)
            img = anonimize(img, x, y, w, h)
            cv2.imwrite(output_directory + "/" + fn + ".png", img,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return

def crop_all_files(input_directory, output_directory, 
                x=CROP_X_DEFAULT, y=CROP_X_DEFAULT, w=CROP_W_DEFAULT, h=CROP_H_DEFAULT):
    ext = ('.jpg', '.jpeg', '.png')
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))
            img = cv2.imread(input_directory + "/" + fname)
            img = img[y:y+h, x:x+w]
            cv2.imwrite(output_directory + "/" + fn + ".png", img,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return


def transform_all_annotations(input_directory, output_directory):
    ext = ('.png')
    i = 0
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))
            fpath = os.path.join(input_directory, fname)
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print("before value conversion:")
            # for i in range(256):
            #     n = np.sum(img == i)
            #     print("number of {}={}".format(i, n))
            img_h, img_w = img.shape
            mask = np.empty((img_h, img_w, 1), dtype = "uint8")
            mask[:] = VALUE_BACKGROUND
            mask[img >= 127] = VALUE_FOREGROUND
            # print("after value conversion:")
            # for i in range(256):
            #     n = np.sum(mask == i)
            #     print("number of {}={}".format(i, n))
            fpath = os.path.join(output_directory, fn + ".png")
            cv2.imwrite(fpath, mask,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if i % 50 == 0:
            print (".", end = "", flush=True)
        i += 1
    print("\n")
    return


def transform_all_image_samples(input_directory, output_directory):
    ext = ('.jpg')
    i = 0
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))
            # read file and decode image
            fpath = os.path.join(input_directory, fname)
            img0 = tf.io.read_file(fpath)
            img0 = tf.image.decode_jpeg(img0, channels=3)
            # cast
            # img0 = tf.cast(img0, tf.float32) / 255.0    # normalize
            img1 = tf.cast(img0, tf.uint8)
            # encode and write
            img1 = tf.image.encode_jpeg(img1)
            fpath = os.path.join(output_directory, fn + ".jpg")
            tf.io.write_file(fpath, img1)
        if i % 50 == 0:
            print (".", end = "", flush=True)
        i += 1
    print("\n")
    return


################
#  main
################
print("Creating images subdir {}...".format(IMAGES_SUBDIR))
os.mkdir(IMAGES_SUBDIR)
print("Converting all sample images...")
transform_all_image_samples(IMAGES_ORIG, IMAGES_SUBDIR)

print("Creating images subdir {}...".format(ANNOTATIONS_SUBDIR))
os.mkdir(ANNOTATIONS_SUBDIR)
print("Converting all annotation images...")
transform_all_annotations(ANNOTATIONS_ORIG, ANNOTATIONS_SUBDIR)

print("Program ended.")
