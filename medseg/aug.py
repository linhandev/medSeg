# 对数据进行增强
# 1. 需要能处理numpy数组，不只是图片
# 2. 要能处理2d和3d的情况
# 3. 所有的操作执行不执行要概率控制
# 4. 首先实现一个能用的版本，逐步实现都用基础矩阵操作不调包
# 5. 所有的函数的默认参数都是调用不做任何变化

import random
import math
import time

import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
import matplotlib.pyplot as plt
import skimage.io

from utils.util import pad_volume

random.seed(time.time())


def flip(volume, label=None, chance=(0, 0, 0)):
    """.

    Parameters
    ----------
    volume : type
        Description of parameter `volume`.
    label : type
        Description of parameter `label`.
    chance : type
        Description of parameter `chance`.

    Returns
    -------
    flip(volume, label=None,
        Description of returned object.

    """
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
def rotate(volume, label=None, angel=([0, 0], [0, 0], [0, 0]), chance=(0, 0, 0), cval=0):
    """ 按照指定象限旋转
    angel：是角度不是弧度
    """

    for axes in range(3):
        if random.random() < chance[axes]:
            rand_ang = angel[axes][0] + random.random() * (angel[axes][1] - angel[axes][0])
            volume = scipy.ndimage.rotate(
                volume,
                rand_ang,
                axes=(axes, (axes + 1) % 3),
                reshape=False,
                mode="constant",
                cval=cval,
            )
            if label is not None:
                label = scipy.ndimage.rotate(
                    label, rand_ang, axes=(axes, (axes + 1) % 3), reshape=False,
                )
    if label is not None:
        return volume, label
    return volume


# 缩放大小， vol 是三阶， lab 是插值， 给的是目标大小
def zoom(volume, label=None, ratio=[(1, 1), (1, 1), (1, 1)], chance=(0, 0, 0)):
    ratio = list(ratio)
    chance = list(chance)
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


def crop(volume, label=None, size=[3, 512, 512], pad_value=0):
    """在随机位置裁剪出一个指定大小的体积
    每个维度都有输入图片更大或者size更大两种情况：
    - 如果输入图片更大，保证不会裁剪出图片，位置随机;
    - 如果size更大，只进行pad操作，体积在正中间.
    对于标签，标签中是1的维度不会进行pad；不是1的和volume都一样
    Parameters
    ----------
    volume : np.ndarray
        Description of parameter `volume`.
    label : np.ndarray
        Description of parameter `label`.
    size : 需要裁出的体积，list
        Description of parameter `size`.
    pad_value : int
        volume用pad_value填充，标签默认用0填充.

    Returns
    -------
    type
        size大小的ndarray.

    """
    # 1. 先pad一手，让数据至少size大
    volume = pad_volume(volume, size, pad_value, False)
    if label is not None:
        lab_size = list(size)
        for ind, s in enumerate(label.shape):
            if s == 1:  # 是1的维度都不动
                lab_size[ind] = -1
        label = pad_volume(label, lab_size, 0, False)
    # 2.随机一个裁剪范围起点，之后进行crop裁剪
    crop_low = [int(random.random() * (x - y)) for x, y in zip(volume.shape, size)]
    r = [[l, l + s] for l, s in zip(crop_low, size)]
    volume = volume[r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]]
    if label is not None:
        for ind in range(3):
            if label.shape[ind] == 1:
                r[ind][0] = 0
                r[ind][1] = 1
        label = label[r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]]
        return volume, label
    return volume


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    print("randomstate", random_state.rand(*shape) * 2 - 1, "end")
    dx = (
        gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    )
    print(dx.shape)
    dy = (
        gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    )
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode="reflect")
    return distored_image.reshape(image.shape)


# TODO: 增加随机shift的增强，这个针对显影剂，随机给整个输入加上一个值，肝脏是-80到0的正态
# TODO: 增加随机噪音的增强
# TODO: 增加图片随机组合的增强
# TODO: 全部流程放进一个函数

# cat = skimage.io.imread("~/Desktop/cat.jpg")
# checker = skimage.io.imread("~/Desktop/checker.png")
# img = np.ones([10, 10])
#
# img[0:5, 0:5] = 0
#
#
# plt.imshow(cat)
# plt.show()
# if img.ndim == 2:
#     img = img[:, :, np.newaxis]

# print(img.shape)
# img, lab = flip(cat, cat, (1, 1, 0))
# plt.imshow(img)
# plt.show()
#
# img, lab = rotate(cat, cat, ([-45, 45], 0, [0, 0]), (1, 0, 0))
# plt.imshow(img)
# plt.show()
#
# img, lab = zoom(cat, cat, [(0.2, 0.3), (0.7, 0.8), (0.9, 1)], (0.5, 1, 0))
# plt.imshow(img)
# plt.show()

# print(cat.shape)
# img, label = crop(cat, cat, [400, 500, 3])
# plt.imshow(img)
# plt.show()
#
# plt.imshow(checker)
# plt.show()
# checker = checker[:, :, np.newaxis]
# img = crop(cat, None, [512, 512, 3])
# img = elastic_transform(checker, 900, 8)
# img = img.reshape(img.shape[0], img.shape[1])
# plt.imshow(img)
# plt.show()

#
# print(img)
# if img.shape[2] == 1:
#     img = img.reshape(img.shape[0], img.shape[1])
# plt.imshow(img)
# plt.show()

# plt.imshow(lab)
# plt.show()


"""
paddle clas 增广策略

图像变换类：
旋转
色调
背景模糊
透明度变换
饱和度变换

图像裁剪类：
遮挡

图像混叠：
两幅图一定权重直接叠
一张图切一部分放到另一张图
"""
