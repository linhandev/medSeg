# coding=utf-8

from lib.store_info_dict_module import store_info_dict
from mate.preprocess_module import preprocess
import numpy as np

para_dict = {
    'organ_names': ['Lung_L', 'Lung_R'],
    'organ_version': '1.0.0',
    'is_print_logo': True,
    'is_save_csv': False,
    'organ_classify_pos': [7, 8],
    'model_input_size': [320, 320],
    'target_spacing': [1.0, 1.0],
}


def preprocess_lung_part(image_raw, info_dict):
    """进行肺部预处理"""
    info_dict = store_info_dict(info_dict, para_dict)

    # 2.读取数据
    info_dict['loaded_image_shape'] = image_raw.shape

    # 3.做classification
    info_dict.head2feet = False
    info_dict.head_adjust_angle = 0.0
    info_dict.head_adjust_center = (256.0, 256.0)
    info_dict.body_adjust_angle = 0.0
    info_dict.body_adjust_center = (256.0, 256.0)

    if image_raw.shape[0] < 3:
        image_raw = np.repeat(image_raw, 3, 0)
    # 4.数据前处理
    data2D, info_dict = preprocess(image_raw, info_dict)

    del info_dict['orig_images']
    del info_dict['images']
    info_dict['img_centre'] = [int(i) for i in info_dict['img_centre']]

    return data2D, info_dict