# Introduction
In this script we are going to cover the usage of tensorflow 2 and tf.data on a semantic segmentation 2D images dataset.

The type of data we are going to manipulate consist in:

    a png image with 3 channels (RGB)
    a png mask with 1 channel (for each pixel we have 1 true class over 150 possible)

You can find useful information by reading the official tensorflow tutorials:

    https://www.tensorflow.org/tutorials/load_data/images
    https://www.tensorflow.org/tutorials/images/segmentation

The direcories should be organized this way:

    ./dataset/images/training/*.png <-- images used for training
    ./dataset/images/validation/*.png <-- images used for validation
    ./dataset/annotations/training/*.png <-- trimap images for training
    ./dataset/annotations/validation/*.png <-- trimap images for validarion

# Run

```sh
$ python main.py
```

# License

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
