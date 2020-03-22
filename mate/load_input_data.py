# coding=utf-8

from lib.info_dict_module import InfoDict
from lib.load_data_module import load_data
from lib.resize_module import judge_resize_512


def load_input_data(input_dir_path):
    """
    读取dcm数据
    """
    info_dict = InfoDict()
    info_dict.data_path = input_dir_path

    image_raw, info_dict = load_data(info_dict)

    # 保证图像为512x512大小
    image_raw = judge_resize_512(image_raw, 'linear')
    info_dict.image_shape_raw = image_raw.shape

    classes_value = [7] * len(image_raw)
    info_dict.classes_value = classes_value

    return image_raw, info_dict
