#coding=utf-8

from skimage.measure import label


def cal_lesion_num(lesion_tuple):
    lesion_l, lesion_r, lesion_part = lesion_tuple

    lesion_num_dict = {
        'lung_l': label(lesion_l, return_num=True)[1],
        'lung_r': label(lesion_r, return_num=True)[1],
        'lung_all': label(lesion_part, return_num=True)[1]
    }

    return lesion_num_dict