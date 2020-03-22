#coding=utf-8


def cal_percent(lesion_part, lung_part):
    return round((lesion_part.sum() / lung_part.sum()), 6)


def cal_lesion_percent(lung_tuple, lesion_tuple):
    """
    计算病灶占比
    整体流程：
    1. 分别得到左右肺和左右病灶
    2. 分别计算相应的百分比
    3. 存储到字典中
    """
    # 1. 分别得到左右肺和左右病灶
    lung_l, lung_r, lung_part = lung_tuple
    lesion_l, lesion_r, lesion_part = lesion_tuple

    # 2. 分别计算相应的百分比
    percent_l = cal_percent(lesion_l, lung_l)
    percent_r = cal_percent(lesion_r, lung_r)
    percent_all = cal_percent(lesion_part, lung_part)

    # 3. 存储到字典中
    percent_dict = {
        'lung_l': percent_l,
        'lung_r': percent_r,
        'lung_all': percent_all
    }

    return percent_dict
