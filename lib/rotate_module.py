#-*- coding:utf-8 -*-

import cv2
import numpy as np
# =============================================================================
'''图像旋转函数_2D'''


def rotate_2D(image, angle, center=None, scale=1.0, use_adjust=False, borderValue=-1000):
	if not use_adjust:
		return image
	if not angle:
		return image
	(h, w) = image.shape[1:]
	image = image.astype(np.int16)
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=borderValue)
	return rotated


# =============================================================================
'''图像旋转函数_3D'''


def rotate_3D(image_3D, angle, center=None, scale=1.0, use_adjust=False, borderValue=-1000):
	if not use_adjust:
		return image_3D
	if not angle:
		return image_3D
	(h, w) = image_3D.shape[1:]
	image_3D = image_3D.astype(np.int16)
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	
	M = cv2.getRotationMatrix2D(center, angle, scale)
	print(image_3D.shape)
	for i in range(image_3D.shape[0]):
		image_3D[i, :, :] = cv2.warpAffine(
			image_3D[i, :, :], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=borderValue)
	return image_3D


# =============================================================================
'''Label2D旋转函数'''


def rotate_label_2D(image, angle, center=None, scale=1.0, use_adjust=False, borderValue=0):
	if not use_adjust:
		return image
	if not angle:
		return image
	(h, w) = image.shape[1:]
	image = image.astype(np.int16)
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	angle = -angle
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=borderValue)
	return rotated


# =============================================================================
'''Label3D旋转函数'''


def rotate_label_3D(image_3D, angle, center=None, scale=1.0, use_adjust=False, borderValue=0):
	if not use_adjust:
		return image_3D
	if not angle:
		return image_3D
	image_3D = image_3D.astype(np.int16)
	(h, w) = image_3D.shape[1:]
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	angle = -angle
	M = cv2.getRotationMatrix2D(center, angle, scale)
	for i in range(image_3D.shape[0]):
		image_3D[i, :, :] = cv2.warpAffine(
			image_3D[i, :, :], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=borderValue)
	return image_3D