# coding=utf-8

import cv2
import numpy as np


# cv2 插值类型
class CV2_interp_type(object):
    linear = cv2.INTER_LINEAR
    nearest = cv2.INTER_NEAREST
    area = cv2.INTER_AREA
    cubic = cv2.INTER_CUBIC
    lanczos = cv2.INTER_LANCZOS4


def interp_2d_yx (image_2d, row_size_new, col_size_new,kind=CV2_interp_type.linear, kernal_size=0):
    '''2d image interpolation
    :param image_2d: 2d volume, format: yx
    :param row_size_new:   can be call "y"
    :param col_size_new:   can be call "x"
    :param kind: interpolation methods, cv2.INTER_LINEAR cv2.INTER_NEAREST cv2.INTER_CUBIC(slow)
    :param kernel_size: used in median blurring for interpolation results. if 0, then no blurring operation
    :return: resized image volume ,its dtype is same with image_2d
    '''
    resize_slice = cv2.resize(image_2d, (col_size_new,row_size_new), interpolation=kind)
    resize_slice = resize_slice
    if kernal_size:
        # smoothes an image using the median filter
        image_new = cv2.medianBlur(resize_slice, kernal_size)
    else:
        image_new = resize_slice
    image_new = np.array(image_new, dtype=image_2d.dtype)

    return image_new


def interp_2d_zyx(image_3d, row_size_new, col_size_new,kind=CV2_interp_type.linear, kernal_size=0):
    z,y,x = image_3d.shape
    rst_array = np.zeros((z, col_size_new, row_size_new), dtype=image_3d.dtype)
    for i in range(z):
        rst_array[i] = interp_2d_yx(image_3d[i], row_size_new, col_size_new, kind, kernal_size)

    return rst_array


def judge_resize_512(image_3d, kind='linear'):
    z,y,x = image_3d.shape

    if (y == 512) and (x == 512):
        return image_3d

    if kind == 'linear':
        kind = CV2_interp_type.linear
    elif kind == 'near':
        kind = CV2_interp_type.nearest
    else:
        raise ValueError('kind type error')

    rst_array = interp_2d_zyx(image_3d, 512, 512, kind=kind)

    return rst_array


if __name__ == '__main__':
    from lkm_lib.utlis.visualization import plotOneImage
    a = np.zeros((36,300,300))
    a[:,100:200,100:200] = 1
    plotOneImage(a[0])
    b = interp_2d_zyx(a, 512, 512)
    plotOneImage(b[0])
    print(b.shape)