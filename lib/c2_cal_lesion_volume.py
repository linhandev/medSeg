#coding=utf-8


def cal_volume(lesion_part, spacing_list):
    lesion_volume = lesion_part.sum()
    for spacing_value in spacing_list:
        lesion_volume = lesion_volume * spacing_value

    return round(lesion_volume, 6)


def cal_lesion_volume(lesion_tuple, spacing_list):
    """
    计算病灶体积
    1. 得到左右病灶
    2. 分别根据spacing计算体积
    3. 返回体积字典
    """
    # 1. 得到左右病灶
    lesion_l, lesion_r, lesion_part = lesion_tuple

    # 2. 分别根据spacing计算体积
    volume_l = cal_volume(lesion_l, spacing_list)
    volume_r = cal_volume(lesion_r, spacing_list)
    volume_all = cal_volume(lesion_part, spacing_list)

    # 3. 返回体积字典
    lesion_volume_dict = {
        'lung_l': volume_l,
        'lung_r': volume_r,
        'lung_all': volume_all
    }

    return lesion_volume_dict


