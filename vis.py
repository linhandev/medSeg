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
import numpy
import os
from medseg.lib.threshold_function_module import windowlize_image
import matplotlib.pyplot as plt
import medseg.config as config

vol_dir = config.volumes_path
lab_dir = config.labels_path

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
        plt.figure(figsize=(15, 15))
        plt.subplot(121)
        plt.imshow(vol[:, :, sli_ind])
        plt.subplot(122)
        plt.imshow(lab[:, :, sli_ind])
        plt.show()
        plt.clbose()
