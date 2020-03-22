# coding=utf-8

import os
import numpy as np
import tqdm
from lib.threshold_function_module import windowlize_image
from lib.judge_mkdir import judge_mkdir

import cv2


def judge_mkdir(path):
    path = path.strip()
    if not os.path.exists(path):
        os.makedirs(path)


def dcm2uint16Param(img_min, img_max, upper_limit=2 ** 16 - 16):
    a = upper_limit / (img_max - img_min)
    b = upper_limit * img_min / (img_min - img_max)
    return a, b


def npy_to_png(input_array):
    a, b = dcm2uint16Param(np.min(input_array), np.max(input_array))
    rst_array = np.array(input_array * a + b, dtype=np.int32)
    return rst_array


def save_merged_png_cv2(image_raw, lung_part, lesion_part, save_dir, merge_name='merge'):
    if merge_name is not None:
        merge_dir = os.path.join(save_dir, merge_name)
    else:
        merge_dir = save_dir
    judge_mkdir(merge_dir)

    image_raw = windowlize_image(image_raw, 1500, -500)
    tq = tqdm.tqdm(range(len(image_raw)), 'creating the merge png images')
    for i in tq:
        image = image_raw[i]
        image = npy_to_png(image)
        image = (image - float(np.min(image))) / float(np.max(image)) * 255.

        lung = lung_part[i,..., 1] + lung_part[i,..., 2]
        binary = lung * 255
        binary = binary.astype(np.uint8)
        try:
            _, lung_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            lung_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        binary = lesion_part[i] * 255
        binary = binary.astype(np.uint8)

        try:
            _, lesion_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            lesion_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image = image[np.newaxis, :, :]
        image = image.transpose((1, 2, 0)).astype('float32')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, lesion_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(image, lung_contours, -1, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(merge_dir, 'layer%d.png' % i), image)