# 将mhd的扫描转换为nii格式
import util
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

scan_dir = "/home/aistudio/sliver/scan"
label_dir = "/home/aistudio/sliver/label"

scan_out = "/home/aistudio/data/scan_temp"
label_out = "/home/aistudio/data/label_temp"


for fname in tqdm(util.listdir(scan_dir)):
    if fname.endswith(".raw"):
        continue
    scan = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, fname)))
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, fname.replace("orig", "seg"))))

    scan = scan.swapaxes(0, 1).swapaxes(1, 2)
    label = label.swapaxes(0, 1).swapaxes(1, 2)

    # label *= 255
    # plt.imshow(label[:, :, 100])
    # plt.show()
    # plt.imshow(scan[:, :, 100])
    # plt.show()

    new_scan = nib.Nifti1Image(scan, np.eye(4))
    new_label = nib.Nifti1Image(label, np.eye(4))

    nib.save(new_scan, os.path.join(scan_out, fname.replace("mhd", "nii").replace("liver-orig", "sliver")))
    nib.save(new_label, os.path.join(label_out, fname.replace("mhd", "nii").replace("liver-orig", "sliver")))
