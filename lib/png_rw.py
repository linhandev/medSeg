#coding=utf-8

import os
import numpy as np
import tqdm
from lib.threshold_function_module import windowlize_image
from lib.judge_mkdir import judge_mkdir

import cv2

def dcm2uint16Param(img_min, img_max, upper_limit=2 ** 16 - 16):
    a = upper_limit / (img_max - img_min)
    b = upper_limit * img_min / (img_min - img_max)
    return a, b


def npy_to_png(input_array):
    a, b = dcm2uint16Param(np.min(input_array), np.max(input_array))
    rst_array = np.array(input_array * a + b, dtype=np.int32)
    return rst_array


def save_raw_png_cv2(image_raw, save_dir, raw_name='raw'):
    if raw_name is not None:
        raw_dir = os.path.join(save_dir, raw_name)
    else:
        raw_dir = save_dir

    judge_mkdir(raw_dir)

    image_raw = windowlize_image(image_raw, 1500, -500)
    tq = tqdm.tqdm(range(len(image_raw)), 'creating the raw png images')
    for i in tq:
        image = image_raw[i]
        image = npy_to_png(image)
        image = (image - float(np.min(image))) / float(np.max(image)) * 255.

        image = image[np.newaxis, :, :]
        image = image.transpose((1, 2, 0)).astype('float32')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(raw_dir, 'layer%d.png' % i), image)