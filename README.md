# Introduction
This script implements a Convolutional Neural Network (U-net) to perform the semantic segmentation of 2D images. The framework used is Keras by Tensorflow.

The type of data that we are going to manipulate consists in:

    - a jpeg image with 3 channels (RGB)
    - a png mask with 1 channel (for each pixel we have 1 true class over 150 possible)

You can find useful information by reading the official tensorflow tutorials:

    https://www.tensorflow.org/tutorials/load_data/images
    https://www.tensorflow.org/tutorials/images/segmentation

The directories that contain the images of the dataset must be organized in the following hierarchy, where training and corresponding validating images must have the same filename:

```sh
    dataset/
        |
        |- images/
        |        |
        |        |- training/
        |        |      |- *.jpg <-- images used for training
        |        |      
        |        |- validation/
        |               |- *.jpg <-- images used for validation
        |
        |- annotations/
                 |
                 |- training/
                 |      |- *.png <-- trimap images used for training
                 |      
                 |- validation/
                        |- *.png <-- trimap images used for validation
```

The actions that can be performed by the script are:
- split;
- train;
- predict;
- evaluate;
- summary;
- inspect.

# Setup Environment
The program has been tested in the following environment:
- Windows 10
- Python 3.8.0
- Tensorflow 2.3.0

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
```

This is the help message that you will get:
```sh
usage: main.py [-h] [--version] [--check] [-ir INITIAL_ROOT_DIR]
               [-dr DATASET_ROOT_DIR] [-w WEIGTHS_FILE] [-i INPUT_IMAGE]
               [-o OUTPUT_FILE] [-s train_p validation_p test_p] [-e EPOCHS]
               [-b BATCH_SIZE] [-m {unet,deeplabv3plus}]
               {split,train,predict,summary,evaluate}

Copyright (C) 2022 Giansalvo Gusinu

positional arguments:
  {split,train,predict,summary,evaluate}
                        The action to perform: split, train, predict, summary, evaluate

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --check               Display some images from dataset before training to check that dataset is ok.
  -ir INITIAL_ROOT_DIR, --initial_root_dir INITIAL_ROOT_DIR
                        The initial root directory of images and trimaps.
  -dr DATASET_ROOT_DIR, --dataset_root_dir DATASET_ROOT_DIR
                        The root directory of the dataset.
  -w WEIGTHS_FILE, --weigths_file WEIGTHS_FILE
                        The weigths file to be loaded/saved. It must be compatible with the network model chosen.
  -i INPUT_IMAGE, --input_image INPUT_IMAGE
                        The input file to be segmented.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        The output file with the segmented image.
  -s train_p validation_p test_p, --split_percentage train_p validation_p test_p
                        The percentage of images to be copied respectively to train/validation/test set.
  -e EPOCHS, --epochs EPOCHS
                        The number of times that the entire dataset is passed forward and backward through the network during the training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        the number of samples that are passed to the network at once during the training
  -m {unet,deeplabv3plus}, --model {unet,deeplabv3plus}
                        The model of network to be created/used. It must be compatible with the weigths file.
```

## Examples

Prepare the dataset directories hierarchy starting from images/annotations initial directories:
```sh
    $python main.py split -ir initial_root_dir -dr dataset_root_dir
    $python main.py split -ir initial_root_dir -dr dataset_root_dir -s 0.4 0.3 0.3
```

Show a summary of the network's structure to the standard output and save the model to disk:
```sh
    $python main.py summary -m deeplabv3plus
```

Train the network:
```sh
    $python main.py train -m deeplabv3plus
    $python main.py train -m deeplabv3plus -dr dataset_dir -w weigths_file.h5
    $python main.py train -m deeplabv3plus -dr dataset_dir -w weigths_file.h5 --check
```

Make the network predict a segmented image given an input image and the weights file:
```sh
    $python main.py predict -m deeplabv3plus -i image.jpg
    $python main.py predict -i image.jpg -w weigths_file.h5 -o image_segm.jpg --check
```

Evaluate the network loss/accuracy performances based on the test set in the dataset directories hierarchy:
```sh
    $python main.py evaluate -m deeplabv3plus -w weigths_file.h5 -dr dataset_dir --check
```

Inspect some predictions obtained in a previous training with --save_predictions::
```sh
    $python main.py inspect -r *.png
```

# Credits

Part of the code was adapted from original code and article by Yann Leguilly:
- https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
- https://github.com/dhassault/tf-semantic-example

The implementation of the deeplabv3plus was adapted from code by Emil Zakirov and others:
- https://github.com/bonlime/keras-deeplab-v3-plus

# License

```sh
Copyright (C) 2022 Giansalvo Gusinu
Copyright (c) 2021 Emil Zakirov and others
Copyright (c) 2020 Yann Le Guilly

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