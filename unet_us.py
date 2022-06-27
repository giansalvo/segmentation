# https://www.kaggle.com/code/vidushibhatia/2-ultrasound-nerve-seg-unet-from-scratch


# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy

TRANSF_LEARN_FREEZE_ENCODER = "freeze_encoder"
TRANSF_LEARN_FREEZE_DECODER = "freeze_decoder"
TRANSF_LEARN_FREEZE_MAX     = "freeze_max"

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True, transfer_learning=None):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    if transfer_learning == TRANSF_LEARN_FREEZE_ENCODER or transfer_learning == TRANSF_LEARN_FREEZE_MAX:
        trainable = False
    else:
        trainable = True

    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv1 = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')
    conv1.trainable = trainable
    conv1 = conv1(inputs)

    conv2 = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')
    conv2.trainable = trainable
    conv2 = conv2(conv1)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv2, training=False)

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

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, transfer_learning=None):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    if transfer_learning == TRANSF_LEARN_FREEZE_DECODER or transfer_learning == TRANSF_LEARN_FREEZE_MAX:
        trainable = False
    else:
        trainable = True

    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')
    up.trainable = trainable
    up = up(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    merge.trainable = trainable
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv1 = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')
    conv1.trainable = trainable
    conv1 = conv1(merge)                 

    conv2 = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')
    conv2.trainable = trainable                 
    conv2 = conv2(conv1)
    return conv2

def create_model_UNet_US(input_size=(128, 128, 3), n_filters=32, n_classes=2, transfer_learning=None):
    inputs = Input(input_size)

    if transfer_learning == TRANSF_LEARN_FREEZE_ENCODER:
        print("unet_us.py: transfer_learning: freeze encoder")
    elif transfer_learning == TRANSF_LEARN_FREEZE_DECODER:
        print("unet_us.py: transfer_learning: freeze decoder")
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True, transfer_learning=transfer_learning)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True, transfer_learning=transfer_learning)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True, transfer_learning=transfer_learning)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True, transfer_learning=transfer_learning)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False, transfer_learning=transfer_learning)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8, transfer_learning=transfer_learning)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4, transfer_learning=transfer_learning)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2, transfer_learning=transfer_learning)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters, transfer_learning=transfer_learning)

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
    model = tf.keras.Model(inputs=inputs, outputs=conv10, name="U-Net-US")

    return model