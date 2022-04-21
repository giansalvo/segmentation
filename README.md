# Introduction
This script implements a Convolutional Neural Network (U-net) to perform the semantic segmentation of 2D images. The framework used is Keras by Tensorflow.

The type of data we are going to manipulate consist in:

    a jpeg image with 3 channels (RGB)
    a png mask with 1 channel (for each pixel we have 1 true class over 150 possible)

You can find useful information by reading the official tensorflow tutorials:

    https://www.tensorflow.org/tutorials/load_data/images
    https://www.tensorflow.org/tutorials/images/segmentation

The directories that contain the images of the dataset must be organized in the following hierarchy, where training and corresponding validating images must have the same filename:

    ./dataset/images/training/*.jpg <-- images used for training
    ./dataset/images/validation/*.jpg <-- images used for validation
    ./dataset/annotations/training/*.png <-- trimap images for training
    ./dataset/annotations/validation/*.png <-- trimap images for validarion

The actions that can be performed by the script are:
- train;
- predict;
- summary.

# Setup Environment

## Linux and MacOS
```sh
$ python -m venv env
$ source env/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

## Windows
```sh
> python -m venv env
> .\env\Scripts\activate
> python -m pip install --upgrade pip
> pip install -r requirements.txt
```

# Sintax
Call the script with the appropriate parameters. To get help from the script just type:
```sh
$ python main.py -h

usage: main.py [-h] [--version] [-v | -q] [--check] [-r DATASET_ROOT_DIR]
               [-w WEIGTHS_FILE] [-i INPUT_IMAGE] [-o OUTPUT_FILE]
               {train,predict,summary}

Copyright (C) 2022 Giansalvo Gusinu <profgusinu@gmail.com>

positional arguments:
  {train,predict,summary}
                        The action to perform: train, predict, summary

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose
  -q, --quiet
  --check               Display some images from dataset before training to check that dataset is ok
  -r DATASET_ROOT_DIR, --dataset_root_dir DATASET_ROOT_DIR
                        The root directory for images
  -w WEIGTHS_FILE, --weigths_file WEIGTHS_FILE
                        The weigths file to be loaded/saved
  -i INPUT_IMAGE, --input_image INPUT_IMAGE
                        The input file to be segmented
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        The output file with the segmented image
```

## Examples

Create the network and show a summary of the network's structure to the standard output:
```sh
$ python main.py summary
```

Train the network:
```sh
$ $python main.py train

$ $python main.py train -r dataset_dir -w weigths_file.h5

$ python main.py train -r dataset_dir -w weigths_file.h5 --check
```

Make the network predict a segmented image from a given input image and the weights:
```sh
$ python main.py predict -i image.jpg

$ python main.py predict -i image.jpg -w weigths_file.h5 -o image_segm.jpg
```
# Credits

Project inspired and adapted from original code and article by Yann Leguilly:
- https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
- https://github.com/dhassault/tf-semantic-example

# License

```sh
Copyright (C) 2022 Giansalvo Gusinu
Copyright (c) 2020 Yann LE GUILLY

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
```