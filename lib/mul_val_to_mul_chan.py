#coding=utf-8

import numpy as np


def _to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    功能: keras官方的转换多channel函数
    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def mul_val_to_mul_chan(input_array, is_save_background=True):
    """
    功能：多数值的数组转换为多channel
    注意：1. 如果返回数组只想保留有数值的层，去除背景层，
            则应该设置is_save_background为False
    """
    rst_array = _to_categorical(input_array)

    if not is_save_background:
        rst_array = rst_array[...,1:]

    return rst_array

