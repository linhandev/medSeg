# coding=utf-8

import tqdm
import numpy as np
from lib.opening_module import opening
from lib.remove_small_holes_module import remove_small_holes
from lib.keep_largest_connection_module import keep_largest_connection
from lib.threshold_function_module import remove_metal_artifact


def process_lung_part(lung_part):
    """
    处理肺部区域
    1. 分开左右肺
    2. 进行闭操作
    3. 进行去除空洞
    4. 保留最大联通区域
    """
    # 1. 分开左右肺
    lung_l = lung_part[..., 1]
    lung_r = lung_part[..., 2]

    tq = tqdm.tqdm(range(len(lung_part)), 'process lung part post process')
    for i in tq:
        tmp_l = lung_l[i]
        tmp_r = lung_r[i]
        # 2. 进行闭操作
        tmp_l = opening(tmp_l, 2)
        tmp_r = opening(tmp_r, 2)
        # 3. 进行去除空洞
        tmp_l = remove_small_holes(tmp_l)
        tmp_r = remove_small_holes(tmp_r)

        lung_l[i] = tmp_l
        lung_r[i] = tmp_r

    # 4. 保留最大联通区域
    lung_l = keep_largest_connection(lung_l)
    lung_r = keep_largest_connection(lung_r)

    lung_part[...,1] = lung_l
    lung_part[...,2] = lung_r

    return lung_part


def cut_out_range(lung_part, lesion_part):
    """裁剪掉越界的病灶"""
    lung_sum = np.sum(lung_part[...,1:], axis=-1)
    lesion_part = lesion_part * lung_sum
    lesion_part.astype(np.uint8)

    return lesion_part


def merge_process(image_raw, lung_part, lesion_part):
    """
    对原图，肺的分割，以及病灶分割进行后处理
    整体流程：
    1. 对肺部进行开操作，去空洞，保留最大联通区域处理
    2. 把病灶分割越界的部分去除
    """
    # image_raw = remove_metal_artifact(image_raw)
    # 1. 对肺部进行开操作，去空洞，保留最大联通区域处理
    lung_part = process_lung_part(lung_part)

    # 2. 把病灶分割越界的部分去除
    lesion_part = cut_out_range(lung_part, lesion_part)

    return lung_part, lesion_part
