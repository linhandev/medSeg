# coding=utf-8

import numpy as np
from skimage.measure import label
from skimage.morphology import  remove_small_objects

##窗宽窗位调整
def adjust_ww_wl(image, ww = 250, wl = 250):
    """
    调整图像得窗宽窗位
    :param image: 3D图像
    :param ww: 窗宽
    :param wl: 窗位
    :return: 调整窗宽窗位后的图像
    """
    min = wl - (ww/2)
    max = wl + (ww/2)
    new_image = np.clip(image, min, max)#np.copy(image)
    # new_image[new_image > max] = max
    # new_image[new_image < min] = min
    return new_image

def remove_mask_noises(mask_multi_label, min_size = 1000):
    '''
    消除二值图中的小面积物体
    :param mask_multi_label: ndarray
    :param min_size: scalar
    :return:
    '''
    # from lkm_lib.utlis.visualization import plot2Image, plotOneImage
    unique_values = np.unique(mask_multi_label)[1:]
    img_new = np.zeros(mask_multi_label.shape)
    for v in unique_values:
        # if np.sum(np.array(mask_multi_label == v)) < min_size:
        #     continue
        # else:
        tensor_label, nb_labels = label(np.array(mask_multi_label == v, dtype=np.int16), return_num = True)
        if nb_labels > 1:
            mask_post_remove = remove_small_objects(tensor_label, min_size=min_size, connectivity=2)
        else:
            mask_post_remove = tensor_label
        # plot2Image(tensor_label, mask_post_remove)
        mask_post_remove = np.array(mask_post_remove>0,dtype=np.int16) * v
        img_new += mask_post_remove
    # plotOneImage(img_new)
    return np.array(img_new, dtype=mask_multi_label.dtype)


