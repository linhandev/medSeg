import os
import multiprocessing
import concurrent

import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import argparse
from tqdm import tqdm


from util import to_pinyin
import util


parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, required=True)
parser.add_argument("--seg_dir", type=str, required=True, default="./seg")
parser.add_argument("--png_dir", type=str, required=True, default="./img")
parser.add_argument("--filter", type=bool, default=False)
args = parser.parse_args()


def main():
    # 检查文件匹配情况
    img_names = os.listdir(args.png_dir)
    img_names = [n for n in img_names if n.endswith("mask.png")]
    patient_names = []
    for n in img_names:
        n = n.split("-")
        if n[0] not in patient_names:
            patient_names.append(n[0])
    patient_names = [n + ".nii.gz" for n in patient_names]

    nii_names = os.listdir(args.scan_dir)
    for p in patient_names:
        if p not in nii_names:
            print(p, "dont have nii")
    for n in nii_names:
        if n not in patient_names:
            print(n, "dont have mask")
    print(patient_names)
    input("Press enter to continue")
    if not os.path.exists(args.seg_dir):
        os.makedirs(args.seg_dir)  # TODO: recursive makedir

    pbar = tqdm(patient_names)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    )
    for patient in pbar:
        pbar.set_description(patient)
        if os.path.exists(os.path.join(args.seg_dir, patient)):
            print(patient, "already finished, skipping")
            continue

        patient_imgs = [
            n
            for n in img_names
            if n.split("-")[0] == patient.split(".")[0] and n.endswith("mask.png")
        ]
        patient_imgs.sort(key=lambda n: int(n.split("-")[1].split("_")[0]))
        print(patient, patient_imgs, len(patient_imgs))
        img_data = np.zeros([512, 512, len(patient_imgs) + 2])

        try:
            print(os.path.join(args.scan_dir, patient))
            scanf = nib.load(os.path.join(args.scan_dir, patient))
            scan_header = scanf.header
        except:
            print("!!!!!!!", patient, "error")
            continue
            # scanf = nib.load(os.path.join(args.scan_dir, "张金华_20201024213424575a.nii"))
            # scan_header = scanf.header

        for img_name in patient_imgs:
            img = cv2.imread(os.path.join(args.png_dir, img_name))
            ind = int(img_name.split("-")[1].split("_")[0])
            # img = img.swapaxes(0, 1)
            if "\u4e00" <= patient[0] <= "\u9fff":
                img = img[:, :, 0]
            elif len(patient) > 5:
                img = img[:, :, 0]
            else:
                img = img[:, :, 0]
            img_data[:, :, ind] = img

        img_data[:, :, -1] = img_data[:, :, -2]
        img_data[:, :, 0] = img_data[:, :, 1]
        if args.filter:
            img_data = util.filter_largest_volume(img_data, mode="hard")
        newf = nib.Nifti1Image(img_data.astype(np.float64), scanf.affine, scan_header)
        executor.submit(save_nii, newf, os.path.join(args.seg_dir, patient))


def save_nii(file, dir):
    nib.save(file, dir)


if __name__ == "__main__":
    main()
