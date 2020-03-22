#coding=utf-8

import cv2
from skimage import morphology


# 形态学cv2相关的形状方法
class MorpShapeCV2:
    rect = cv2.MORPH_RECT  # 矩形
    cross = cv2.MORPH_CROSS  # 菱形
    ellipse = cv2.MORPH_ELLIPSE  # 椭圆


# 形态学skimage相关形状方法
class MorpShapeSKI:
    rect = morphology.rectangle  # 矩形
    square = morphology.square  # 正方形
    disk = morphology.disk  # 平面圆形
    ball = morphology.ball  # 球形
    cube = morphology.cube  # 立方体形
    diamond = morphology.diamond  # 钻石形
    star = morphology.star  # 星形
    octagon = morphology.octagon  # 八角形
    octahedron = morphology.octahedron  # 八面体

