"""
对内存中的ndarray，npz，nii进行可视化
"""
import sys
import os
import argparse
import time

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from utils.config import cfg
import utils.util as util
import train


def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("-c", "--cfg_file", type=str, help="配置文件路径")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)


def show_slice(vol, lab):
    """展示一个2.5D的数据对.

    Parameters
    ----------
    vol : ndarray
        2.5D的扫描slice.
    lab : ndarray
        1片分割标签.
    """
    if vol.shape[0] <= 3:  # CWH 需要转换WHC
        vol = vol.swapaxes(0, 2)
        lab = lab.swapaxes(0, 2)
    if lab.ndim == 2:
        lab = lab[:, :, np.newaxis]
    if len(vol.shape) == 3 and vol.shape[2] == 3:  # 如果输入是3 channel的取中间一片
        vol = vol[:, :, 1]
    if len(vol.shape) == 2:
        vol = vol[:, :, np.newaxis]

    vol = np.tile(vol, (1, 1, 3))
    lab = np.tile(lab, (1, 1, 3))
    print("vis shape", vol.shape, lab.shape)
    vmax = vol.max()
    vmin = vol.min()
    vol = (vol - vmin) / (vmax - vmin) * 255
    lab = lab * 255

    vol = vol.astype("uint8")
    lab = lab.astype("uint8")

    plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.imshow(vol)
    plt.subplot(122)
    plt.imshow(lab)
    plt.show()
    plt.close()


def show_nii():
    scans = util.listdir(cfg.DATA.INPUTS_PATH)
    labels = util.listdir(cfg.DATA.LABELS_PATH)
    records = []
    for ind in range(len(scans)):
        # for ind in range(3):
        print(scans[ind], labels[ind])

        scanf = nib.load(os.path.join(cfg.DATA.INPUTS_PATH, scans[ind]))
        labelf = nib.load(os.path.join(cfg.DATA.LABELS_PATH, labels[ind]))
        scan = scanf.get_fdata()
        scan = util.windowlize_image(scan, cfg.PREP.WWWC)
        label = labelf.get_fdata()
        scan, label = util.cal_direction(scans[ind], scan, label)
        print(scan.shape)
        print(label.shape)
        sli_ind = int(scan.shape[2] / 6)
        # for sli_ind in range(vol.shape[2]):
        show_slice(scan[:, :, sli_ind * 2], label[:, :, sli_ind * 2])
        show_slice(scan[:, :, sli_ind * 3], label[:, :, sli_ind * 3])
        show_slice(scan[:, :, sli_ind * 4], label[:, :, sli_ind * 4])
        t = input("是否左右翻转: ")
        records.append([scans[ind], t])
        time.sleep(1)
        print(records)

    f = open("./flip.csv", "w")
    for record in records:
        print(record[0] + "," + record[1], file=f)
    f.close()
    # 1 rot 1 次，2 rot 3 次
    # 0 左右不 flip， 1 左右 flip


def show_npz():
    """对训练数据npz进行可视化.

    """
    for npz in os.listdir(cfg.TRAIN.DATA_PATH):
        data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz))
        vol = data["imgs"]
        lab = data["labs"]
        # for ind in range(vol.shape[0]):
        #     show_slice(vol[ind], lab[ind])
        show_slice(vol[0], lab[0])
        show_slice(vol[vol.shape[0] - 1], lab[vol.shape[0] - 1])


def show_aug():
    """在读取npz基础上做aug之后展示.

    """
    for npz in os.listdir(cfg.TRAIN.DATA_PATH):
        data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz))
        vol = data["imgs"]
        lab = data["labs"]
        vol = vol.astype("float32")
        lab = lab.astype("int32")
        if cfg.AUG.WINDOWLIZE:
            vol = util.windowlize_image(vol, cfg.AUG.WWWC)  # 肝脏常用
        # for ind in range(vol.shape[0]):
        vol_slice, lab_slice = train.aug_mapper([vol[0], lab[0]])
        show_slice(vol_slice, lab_slice)

        vol_slice, lab_slice = train.aug_mapper([vol[vol.shape[0] - 1], lab[vol.shape[0] - 1]])
        show_slice(vol_slice, lab_slice)


if __name__ == "__main__":
    parse_args()
    show_nii()
    # show_npz()
    # show_aug()


# import os
# import matplotlib.pyplot as plt
# from nibabel.orientations import aff2axcodes

# vols = "/home/aistudio/data/volume"
# for voln in os.listdir(vols):
#     print("--------")
#     print(voln)
#
#     volf = sitk.ReadImage(os.path.join(vols, voln))
#
#     vold = sitk.GetArrayFromImage(volf)
#     print(vold.shape)
#     vold[500:512, 250:260, 0] = 2048
#
#     plt.imshow(vold[0, :, :])
#     plt.show()

# vols = "/home/aistudio/data/volume"
# directions = []
# for voln in os.listdir(vols):
#     print("--------")
#     print(voln)
#
#     volf = nib.load(os.path.join(vols, voln))
#     print(volf.affine)
#     print("codes", aff2axcodes(volf.affine))
#
#     vold = volf.get_fdata()
#     vold[500:512, 250:260, 0] = 2048
#
#     plt.imshow(vold[:, :, 0])
#     plt.show()
#
#     cmd = input("direction: ")
#     if cmd == "a":
#         dir = [voln, 1]  # 床在左边
#     else:
#         dir = [voln, 2]  # 床在右边
#     directions.append(dir)
#
#
# f = open("./directions.csv", "w")
# for dir in directions:
#     print(dir[0], ",", dir[1], file=f)
# f.close()


# vols = "/home/aistudio/data/volume"
#
#
# # 获取体位信息
# f = open("./directions.csv")
# dirs = f.readlines()
# print("dirs: ", dirs)
# dirs = [x.rstrip("\n") for x in dirs]
# dirs = [x.split(",") for x in dirs]
# dic = {}
# for dir in dirs:
#     dic[dir[0].strip()] = dir[1].strip()
# f.close()
#
# print(dic)
#
#
# for voln in os.listdir(vols):
#     print("--------")
#     print(voln)
#
#     volf = nib.load(os.path.join(vols, voln))
#     vold = volf.get_fdata()
#     print(dic[voln])
#     if dic[voln] == "2":
#         vold = np.rot90(vold, 3)
#     else:
#         vold = np.rot90(vold, 1)
#
#     plt.imshow(vold[:, :, 50])
#     plt.show()
