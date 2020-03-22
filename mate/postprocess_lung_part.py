# coding=utf-8

import numpy as np
from mate.postprocess_module import postprocess
from lib.resize_module import judge_resize_512
from lib.mul_val_to_mul_chan import mul_val_to_mul_chan


def postprocess_lung_part(lung_part, info_dict):
    """肺部分割结果的后处理"""
    # 7.数据后处理
    label_post, info_dict = postprocess(lung_part, info_dict)

    # 保证图像为512x512大小
    rst_label = judge_resize_512(label_post, 'near')
    # 变为3通道
    rst_label = mul_val_to_mul_chan(rst_label, is_save_background=True).astype(np.uint8)

    return rst_label
