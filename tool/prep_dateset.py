# 将两个文件夹的nii扫描和标签转换成paddleseg的数据集格式
"""
1. 所有的nii和标签转成png
2. 按照paddleseg的目录结构移动文件
3. 生成list文件
"""
import os
import argparse

import numpy as np
import cv2
import nibabel as nib

import util.util as util

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, default="/home/lin/Desktop/data/aorta/nii/scan")
parser.add_argument("--label_dir", type=str, default="/home/lin/Desktop/data/aorta/nii/label")
parser.add_argument("--scan_img_dir", type=str, default="/home/lin/Desktop/data/aorta/dataset/scan")
parser.add_argument("--label_img_dir", type=str, default="/home/lin/Desktop/data/aorta/dataset/label")

args = parser.parse_args()
util.check_nii_match(args.scan_dir, args.label_dir)
util.nii2png_folder(args.scan_dir, args.scan_img_dir, rot=3, wwwl=(600, 0), subfolder=False)
util.nii2png_folder(args.label_dir, args.label_img_dir, rot=3, wwwl=(600, 0), subfolder=False, islabel=True)
