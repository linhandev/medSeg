# coding=utf-8

import numpy as np


def threshold_up_to_up(
        input_array: np.ndarray, threshold: int or float
        ):
    """
    阈值处理,大于threshold等于threshold
    :param input_array: ndarray
    :param threshold: int or float
    :return: ndarray
    """
    #大于threshold等于threshold
    #rst_array = input_array.copy()
    input_array[input_array >= threshold] = threshold
    rst_array = input_array

    return rst_array


def threshold_down_to_down(
        input_array: np.ndarray, threshold: int or float
        ):
    """
    阈值处理,小于threshold等于threshold
    :param input_array: ndarray
    :param threshold: int or float
    :return: ndarray
    """
    # 大于threshold等于threshold
    #rst_array = input_array.copy()
    input_array[input_array <= threshold] = threshold
    rst_array = input_array

    return rst_array


def threshold_updown_to_updown(
        input_array: np.ndarray, up_thre=1000, down_thre=-1000):
    """
    阈值处理,大于threshold等于target
    :param input_array: ndarray
    :param up_thre: int or float
    :param down_thre: int or float
    :return: ndarray
    """
    rst_array = threshold_up_to_up(input_array, up_thre)
    rst_array = threshold_down_to_down(rst_array, down_thre)

    return rst_array


def windowlize_image(input_image:np.ndarray, window_width=None, window_center=None):
    '''
    对传入的CT图像进行窗宽和窗位处理
    :param input_image:  3D image
    :param window_width:  window width
    :param window_center: indow center
    :return: 处理后图像
    '''
    window = [window_center - window_width // 2, window_center + window_width // 2]
    window_image = threshold_updown_to_updown(
        input_image, up_thre=window[1], down_thre=window[0])

    return window_image


def remove_metal_artifact( imgs_array: np.ndarray, max_value=1024,min_value=-1024):
    """
    对传入矩阵进行伪影进行阈值过滤
    :param imgs_array: ndarray
    :param max_value: int or float
    :param min_value: int or float
    :return: imgs_array: ndarray
    """
    #进行阈值处理

    imgs_array = threshold_up_to_up(imgs_array, max_value)
    imgs_array = threshold_down_to_down(imgs_array, min_value)

    return imgs_array
