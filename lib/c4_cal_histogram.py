#coding=utf-8

import numpy as np


def list_to_value_dict(value_list):
    value_dict = dict()
    for value in value_list:
        dict_num = value_dict.get(value, 0)
        value_dict[value] = dict_num + 1

    return value_dict


def merge_two_dict(a_dict, b_dict):
    all_dict = a_dict.copy()
    for key, r_value in b_dict.items():
        l_value = all_dict.get(key, 0)
        all_dict[key] = l_value + r_value

    return all_dict


def cal_histogram(image_input, lung_tuple):
    """
    计算直方图
    整体流程：
    1. 准备处理原图得到肺不同部分
    2. 截取原图中肺的部分
    3. 生成频数字典
    """
    # 1. 准备处理原图得到肺不同部分
    image_raw = image_input.copy()
    image_raw[image_raw > 1024] = 1024
    image_raw[image_raw < -1024] = -1024
    lung_l, lung_r, lung_part = lung_tuple

    # 2. 截取原图中肺的部分
    lung_l_select = image_raw[lung_l == 1]
    lung_r_select = image_raw[lung_r == 1]
    lung_l_select = [int(i) for i in lung_l_select]
    lung_r_select = [int(i) for i in lung_r_select]

    # 3. 生成频数字典
    lung_l_dict = list_to_value_dict(lung_l_select)
    lung_r_dict = list_to_value_dict(lung_r_select)
    lung_all_dict = merge_two_dict(lung_l_dict, lung_r_dict)

    hu_statistics_dict = {
        'lung_l': lung_l_dict,
        'lung_r': lung_r_dict,
        'lung_all': lung_all_dict
    }

    return hu_statistics_dict

