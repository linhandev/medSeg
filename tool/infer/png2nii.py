import os
import os.path as osp
import multiprocessing
import concurrent
from time import sleep
from queue import Queue

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
parser.add_argument("--seg_dir", type=str, required=True)
parser.add_argument("--png_dir", type=str, required=True)
parser.add_argument("--rot", type=int, default=0)
parser.add_argument("--filter", default=False, action="store_true")
args = parser.parse_args()


def get_name(name):
    pos = -name[::-1].find("-") - 1
    return name[:pos], name[pos + 1 :].split(".")[0]


class ThreadPool(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, maxsize=6, *args, **kwargs):
        super(ThreadPool, self).__init__(*args, **kwargs)
        self._work_queue = Queue(maxsize=maxsize)


percent_file = open("./percent.txt", "a+")


def main():
    # 检查文件匹配情况
    img_names = os.listdir(args.png_dir)
    patient_names = []
    for n in img_names:
        # n = n.split("-")
        n, _ = get_name(n)
        if n not in patient_names:
            patient_names.append(n)
    patient_names = [n + ".nii.gz" for n in patient_names]

    nii_names = os.listdir(args.scan_dir)
    for p in patient_names:
        if p not in nii_names:
            print(p, "dont have nii")
    for n in nii_names:
        if n not in patient_names:
            print(n, "dont have mask")
    patient_names.sort()
    print(patient_names)

    input("Press enter to continue！")
    if not os.path.exists(args.seg_dir):
        os.makedirs(args.seg_dir)  # TODO: recursive makedir

    pbar = tqdm(patient_names)
    executor = ThreadPool(max_workers=multiprocessing.cpu_count())

    for patient in pbar:
        pbar.set_description(patient)
        if os.path.exists(os.path.join(args.seg_dir, patient)):
            print(patient, "already finished, skipping")
            continue

        patient_imgs = [n for n in img_names if get_name(n)[0] == patient.split(".")[0]]
        patient_imgs.sort(key=lambda n: int(get_name(n)[1]))
        # print(patient, patient_imgs, len(patient_imgs))
        # TODO: 换成label，不应该叫这个
        img_data = np.zeros([512, 512, len(patient_imgs)], dtype="uint8")

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
            img = cv2.imread(os.path.join(args.png_dir, img_name), cv2.IMREAD_UNCHANGED)
            ind = int(get_name(img_name)[1])
            img_data[:, :, ind] = img

        # img_data[:, :, -1] = img_data[:, :, -2]
        # img_data[:, :, 0] = img_data[:, :, 1]
        # for _ in range(args.rot):
        #     img_data = np.rot90(img_data)
        # if args.filter:
        #     tot = img_data.sum()
        #     img_data = util.filter_largest_volume(img_data, mode="hard")
        #     largest = img_data.sum()
        #     print(patient, largest / tot, file=f)
        #     f.flush()
        # newf = nib.Nifti1Image(img_data.astype(np.float64), scanf.affine, scan_header)

        executor.submit(
            save_nii, img_data, scanf.affine, scan_header, os.path.join(args.seg_dir, patient)
        )
        # save_nii(img_data, scanf.affine, scan_header, os.path.join(args.seg_dir, patient))

    percent_file.close()


def save_nii(label_data, affine, header, dir):
    print("++++", dir)
    for _ in range(args.rot):
        label_data = np.rot90(label_data)
    if args.filter:
        tot = label_data.sum()
        label_data = util.filter_largest_volume(label_data, mode="hard")
        largest = label_data.sum()
        print(osp.basename(dir), largest / tot, file=percent_file)
        percent_file.flush()
    newf = nib.Nifti1Image(label_data.astype(np.float64), affine, header)
    nib.save(newf, dir)
    print("--------", "finish", dir)


if __name__ == "__main__":
    main()
