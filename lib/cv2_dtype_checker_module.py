#coding=utf-8

import numpy as np


def check_for_cv2_dtype(input_array: np.ndarray, raw_dtype):
    """
    判断是否需要改变为cv2支持的dtype
    """
    if raw_dtype in [
        np.int, int, np.int8, np.uint, np.uint32, np.uint64,
        np.int32, np.int64, bool, np.bool]:
        input_array = input_array.astype(np.float32)

    return input_array


def anti_check_for_cv2_dtype(input_array: np.ndarray, raw_dtype):
    """
    判断是否需要反改变为原数据类型
    """
    if raw_dtype in [
        np.int, int, np.int8, np.uint, np.uint32, np.uint64,
        np.int32, np.int64, bool, np.bool]:
        input_array = input_array.astype(raw_dtype)

    return input_array
