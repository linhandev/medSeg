# 对数据进行增强
# 1. 需要能处理numpy数组，不只是图片
# 2. 要能处理2d和3d的情况
# 3. 所有的操作执行不执行要概率控制
# 4. 首先实现一个能用的版本，逐步实现都用基础矩阵操作不调包
# 5. 所有的函数的默认参数都是调用不做任何变化

import random
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.io
from util import pad_volume
import math
import time

random.seed(time.time())


def flip(volume, label=None, chance=(0, 0, 0)):
    if random.random() < chance[0]:
        volume = volume[::-1, :, :]
        if label is not None:
            label = label[::-1, :, :]
    if random.random() < chance[1]:
        volume = volume[:, ::-1, :]
        if label is not None:
            label = label[:, ::-1, :]
    if random.random() < chance[2]:
        volume = volume[:, :, ::-1]
        if label is not None:
            label = label[:, :, ::-1]

    if label is not None:
        return volume, label
    return volume


# x,y,z 任意角度旋转，背景填充，mirror，0，extend
# https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab 需要研究
# TODO: 支持 padvalue
def rotate(volume, label=None, angel=([0, 0], [0, 0], [0, 0]), chance=(0, 0, 0)):
    """ 按照指定象限旋转
    angel：是角度不是弧度
    """

    for axes in range(3):
        if random.random() < chance[axes]:
            rand_ang = angel[axes][0] + random.random() * (angel[axes][1] - angel[axes][0])
            volume = scipy.ndimage.rotate(volume, rand_ang, axes=(axes, (axes + 1) % 3), reshape=False)
            if label is not None:
                label = scipy.ndimage.rotate(label, rand_ang, axes=(axes, (axes + 1) % 3), reshape=False)
    if label is not None:
        return volume, label
    return volume


# 缩放大小， vol 是三阶， lab 是插值， 给的是目标大小
def zoom(volume, label=None, ratio=[(1, 1), (1, 1), (1, 1)], chance=(0, 0, 0)):
    for axes in range(3):
        if random.random() < chance[axes]:  # 如果随机超过做zoom的概率，那就是不做缩放
            ratio[axes] = ratio[axes][0] + random.random() * (ratio[axes][1] - ratio[axes][0])
        else:
            ratio[axes] = 1
    volume = scipy.ndimage.zoom(volume, ratio, order=3, mode="constant")
    if label is not None:
        label = scipy.ndimage.zoom(label, ratio, order=3, mode="constant")
        return volume, label
    return volume


def crop(volume, label=None, size=[512, 512, 512], pad_value=0):
    """在随机位置裁剪出一个指定大小的体积，每个维度都有输入图片更大或者size更大两种情况
    如果输入图片更大，保证不会裁剪出图片，位置随机;
    如果size更大，图片放的位置随机，都会被包括进来，空背景用pad_value补全
	"""

    volume = pad_volume(volume, size, pad_value, False)
    if label is not None:
        lab_size = size
        if label.shape[0] == 1:
            lab_size[0] = 0
        label = pad_volume(label, lab_size, pad_value, False)

    center_range = [[math.floor(size[ind] / 2), volume.shape[ind] - math.ceil(size[ind] / 2)] for ind in range(3)]
    center = [ra[0] + int(random.random() * (ra[1] - ra[0])) for ra in center_range]
    crop_range = [[center[ind] - math.floor(size[ind] / 2), center[ind] + math.ceil(size[ind] / 2)] for ind in range(3)]
    r = crop_range
    # print(center_range, center, crop_range)
    volume = volume[r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]]
    if label is not None:
        # TODO: label这里第一个维度可能不一样，目前的思路是如果是2.5d输入，label [1,x,x] 就第一维全返，后两个裁。如果不是就认为是3d的图，正常裁
        print(label.shape)
        if label.shape[0] != 1:
            label = label[r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]]
        else:
            label = label[:, r[1][0] : r[1][1], r[2][0] : r[2][1]]
        return volume, label

    return volume


# TODO: 将全流程放进一个函数


# 随机添加噪声

# cat = skimage.io.imread("./lib/cat.jpeg")
# plt.imshow(cat)
# plt.show()

# print(cat.shape)
# _, cat = flip(cat, cat, (1, 1, 0))
# cat = rotate(cat, None, ([-45, 45], 0, [0, 0]), (1, 0, 0))
# cat = zoom(cat, None, [(0.2, 0.3), (0.7, 0.8), (0.9, 1)], (0.5, 1, 0))
# print(cat.shape)
# cat = crop(cat, None, [500, 500, 3])

# plt.imshow(cat)
# plt.show()
