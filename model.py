import h5py
from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D


def SqueezeNet(nb_classes, inputs=(3, 224, 224)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    @param nb_classes: total number of final categories

    Arguments:
    inputs -- shape of the input images (channel, cols, rows)

    """

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, 7, 7, activation='relu', init='glorot_uniform',
        subsample=(2, 2), border_mode='same', name='conv1')(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = merge(
        [fire2_expand1, fire2_expand2], mode='concat', concat_axis=1)

    fire3_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = merge(
        [fire3_expand1, fire3_expand2], mode='concat', concat_axis=1)

    fire4_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = merge(
        [fire4_expand1, fire4_expand2], mode='concat', concat_axis=1)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = merge(
        [fire5_expand1, fire5_expand2], mode='concat', concat_axis=1)

    fire6_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = merge(
        [fire6_expand1, fire6_expand2], mode='concat', concat_axis=1)

    fire7_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand2')(fire7_squeeze)
    merge7 = merge(
        [fire7_expand1, fire7_expand2], mode='concat', concat_axis=1)

    fire8_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand2')(fire8_squeeze)
    merge8 = merge(
        [fire8_expand1, fire8_expand2], mode='concat', concat_axis=1)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

    fire9_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_squeeze')(maxpool8)
    fire9_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand2')(fire9_squeeze)
    merge9 = merge(
        [fire9_expand1, fire9_expand2], mode='concat', concat_axis=1)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, 1, 1, init='glorot_uniform',
        border_mode='valid', name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    return Model(input=input_img, output=softmax)
