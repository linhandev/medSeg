# %matplotlib inline
"""
对nii中的切片进行可视化和对保存成2.5d的预处理数据进行可视化
"""
import sys

if "/home/aistudio/external-libraries" not in sys.path:
    sys.path.append("/home/aistudio/external-libraries")
if "/home/aistudio/work" not in sys.path:
    sys.path.append("/home/aistudio/work")
import nibabel as nib
import numpy as np
import os
from lib.threshold_function_module import windowlize_image
import matplotlib.pyplot as plt
import config

vol_dir = config.volumes_path
lab_dir = config.labels_path


def show(vol, lab):
    if vol.shape[0] <= 3:  # CWH 需要转换WHC
        vol = vol.swapaxes(0, 2)
        lab = lab.swapaxes(0, 2)

    if len(vol.shape) == 3 and vol.shape[2] == 3:  # 如果输入是3 channel的取中间一片
        vol = vol[:, :, 1]
    if len(vol.shape) == 2:
        vol = vol[:, :, np.newaxis]

    vol = np.tile(vol, (1, 1, 3))
    lab = np.tile(lab, (1, 1, 3))

    vol = (vol + 300) / 600 * 255
    vol = vol.clip(0, 255)
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
        vol = windowlize_image(vol, 400, 20)
        lab = labf.get_fdata()
        print(vol.shape)
        print(lab.shape)
        # vol = vol.swapaxes(0,2)
        for sli_ind in range(vol.shape[2]):
            show(vol[:, :, sli_ind], lab[:, :, sli_ind])


def show_npy():
    for data_name in os.listdir(config.preprocess_path):
        data = np.load(os.path.join(config.preprocess_path, data_name))
        print(data.shape)
        vol = data[0:3, :, :].reshape(3, 512, 512)
        lab = data[3, :, :].reshape(1, 512, 512)

        show(vol, lab)


if __name__ == "__main__":
    # show_nii()
    show_npy()
