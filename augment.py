"""
Copyright (C) 2022 Giansalvo Gusinu

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
import os
import time
from xml.etree.ElementTree import VERSION
import cv2
import numpy as np
import imutils
import random
import datetime
import argparse

PROGRAM_VERSION = "1.0"
COPYRIGHT_NOTICE = "Copyright(c) 2022 Giansalvo Gusinu"

ACTION_ROTATE = "rotate"
ACTION_TRASLATE = "traslate"

FOLDER_IMAGES = "images"
FOLDER_ANNOTATIONS = "annotations"

ROTATION_DEGREE = 30
TRASLATION_PERCENTAGE = 0.25


def augment_traslate(folder_root_orig, folder_root_dest):
    ext = ('.jpg')
    i = 0
    folder_images_input = os.path.join(folder_root_orig, FOLDER_IMAGES)
    folder_images_output = os.path.join(folder_root_dest, FOLDER_IMAGES)
    folder_annotations_input = os.path.join(folder_root_orig, FOLDER_ANNOTATIONS)
    folder_annotations_output = os.path.join(folder_root_dest, FOLDER_ANNOTATIONS)

    print("Creating folder for augmented images {}...".format(folder_images_output))
    os.mkdir(folder_images_output)
    print("Creating folder for augmented annotations {}...".format(folder_annotations_output))
    os.mkdir(folder_annotations_output)

    for fname in os.listdir(folder_images_input):
        if fname.endswith(ext):
            ### AUGMENTING IMAGE
            fn, fext = os.path.splitext(os.path.basename(fname))
            fin = os.path.join(folder_images_input, fname)
            # read file and decode image
            image = cv2.imread(fin)
            h, w, _ = image.shape
            delta_h = TRASLATION_PERCENTAGE * h * random.uniform(-1, 1)
            delta_w = TRASLATION_PERCENTAGE * w * random.uniform(-1, 1)
            shifted = imutils.translate(image, delta_w, delta_h)
            # cv2.imshow("Shifted Down", shifted)
            # cv2.waitKey(0)
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            fimage = fn + "_" + timestamp + fext
            fout = os.path.join(folder_images_output, fimage)
            cv2.imwrite(fout, shifted)

            ### AUGMENTING ANNOTATION
            fin = os.path.join(folder_annotations_input, fn + ".png")
            # print(str(fin))
            # read file and decode image
            image = cv2.imread(fin)
            shifted = imutils.translate(image, delta_w, delta_h)
            fn = fn + "_" + timestamp + ".png"
            fout = os.path.join(folder_annotations_output, fn)
            cv2.imwrite(fout, shifted)


        if (i % 10) == 0:
            print (".", end = "", flush=True)
        i += 1
    print("\n")
    return

def augment_rotate(folder_root_orig, folder_root_dest):
    ext = ('.jpg')
    i = 0
    folder_images_input = os.path.join(folder_root_orig, FOLDER_IMAGES)
    folder_images_output = os.path.join(folder_root_dest, FOLDER_IMAGES)
    folder_annotations_input = os.path.join(folder_root_orig, FOLDER_ANNOTATIONS)
    folder_annotations_output = os.path.join(folder_root_dest, FOLDER_ANNOTATIONS)

    print("Creating folder for augmented images {}...".format(folder_images_output))
    os.mkdir(folder_images_output)
    print("Creating folder for augmented annotations {}...".format(folder_annotations_output))
    os.mkdir(folder_annotations_output)

    for fname in os.listdir(folder_images_input):
        if fname.endswith(ext):
            ### AUGMENTING IMAGE
            fn, fext = os.path.splitext(os.path.basename(fname))
            fin = os.path.join(folder_images_input, fname)
            # read file and decode image
            image = cv2.imread(fin)
            # h, w, _ = image.shape
            angle = ROTATION_DEGREE * random.uniform(-1, 1)
            rotated = imutils.rotate(image, angle)
            # cv2.imshow("Rotated (Problematic)", rotated)
            # cv2.waitKey(0)
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            fimage = fn + "_" + timestamp + fext
            fout = os.path.join(folder_images_output, fimage)
            cv2.imwrite(fout, rotated)

            ### AUGMENTING ANNOTATION
            fin = os.path.join(folder_annotations_input, fn + ".png")
            # read file and decode image
            image = cv2.imread(fin)
            rotated = imutils.rotate(image, angle)
            fn = fn + "_" + timestamp + ".png"
            fout = os.path.join(folder_annotations_output, fn)
            cv2.imwrite(fout, rotated)


        if (i % 10) == 0:
            print (".", end = "", flush=True)
        i += 1
    print("\n")
    return


################
#  main
################
def main():
    parser = argparse.ArgumentParser(
        description=COPYRIGHT_NOTICE,
        epilog = "Examples:\n"
                "       Prepare the augmented dataset:\n"
                "         $python %(prog)s -a rotate|translate -i input_dataset_folder -o output_dataset_folder\n"
                "\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    parser.add_argument('-i', '--input_dataset_dir', required=True, help="The dataset origin folder.")
    parser.add_argument('-o', '--output_dataset_dir', required=True, help="The dataset destination folder.")
    parser.add_argument('-a', "--action", required=True,
                        choices=(ACTION_ROTATE, ACTION_TRASLATE), 
                        help="The action to augment the dataset: " + ACTION_ROTATE + ", " + ACTION_TRASLATE)
    args = parser.parse_args()

    input_path = args.input_dataset_dir
    output_path = args.output_dataset_dir
    action = args.action

    print("Creating folder for augmented dataset {}...".format(output_path))
    os.mkdir(output_path)
    print("Perform augmentation: " + action)
    if action is ACTION_TRASLATE:
        augment_traslate(input_path, output_path)
    else:
        augment_rotate(input_path, output_path)

    print("Program ended.")

if __name__ == '__main__':
    main()