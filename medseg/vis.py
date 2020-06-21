"""
对内存中的ndarray，npz，nii进行可视化
"""
import sys
import nibabel as nib
import numpy as np
import os
from utils.config import cfg
import utils.util as util
import matplotlib.pyplot as plt
import train


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
    for fname in os.listdir(lab_dir):
        volf = nib.load(os.path.join(vol_dir, fname))
        labf = nib.load(os.path.join(lab_dir, fname))
        vol = volf.get_fdata()
        vol = util.windowlize_image(vol, 400, 20)
        lab = labf.get_fdata()
        print(vol.shape)
        print(lab.shape)
        # vol = vol.swapaxes(0,2)
        for sli_ind in range(vol.shape[2]):
            show_slice(vol[:, :, sli_ind], lab[:, :, sli_ind])


def show_npz():
    """对训练数据npz进行可视化.

    """
    for npz in os.listdir(cfg.TRAIN.DATA_PATH):
        data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz))
        vol = data["vols"]
        lab = data["labs"]
        for ind in range(vol.shape[0]):
            show_slice(vol[ind], lab[ind])


def show_aug():
    """在读取npz基础上做aug之后展示.

    """
    for npz in os.listdir(cfg.TRAIN.DATA_PATH):
        data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz))
        vol = data["vols"]
        lab = data["labs"]
        vol = vol.astype("float32")
        lab = lab.astype("int32")
        if cfg.AUG.WINDOWLIZE:
            vol = util.windowlize_image(vol, cfg.AUG.WWWC)  # 肝脏常用
        for ind in range(vol.shape[0]):
            vol_slice, lab_slice = train.aug_mapper([vol[ind], lab[ind]])
            show_slice(vol_slice, lab_slice)


if __name__ == "__main__":
    # show_nii()
    # show_npz()
    show_aug()
