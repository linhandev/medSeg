"""
将切片的推理结果合起来
默认推理结果是 name-idx.png 格式，格式不同的话修改 get_name 函数
"""
import os
import os.path as osp
import concurrent
from time import sleep
from queue import Queue
import argparse
import multiprocessing
from multiprocessing import Pool

import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from util import to_pinyin
import util


parser = argparse.ArgumentParser()
parser.add_argument(
    "--scan_dir",
    type=str,
    required=True,
    help="扫描路径，会去找头文件信息",
)
parser.add_argument(
    "--seg_dir",
    type=str,
    required=True,
    help="nii分割标签输出路径",
)
parser.add_argument(
    "--png_dir",
    type=str,
    required=True,
    help="png格式分割推理结果路径",
)
parser.add_argument(
    "--rot",
    type=int,
    default=0,
    help="对结果进行几次旋转",
)
parser.add_argument(
    "--filter",
    default=False,
    action="store_true",
    help="是否过滤最大连通块",
)
parser.add_argument(
    "--percent",
    type=str,
    default=None,
    help="最大连通块占所有前景标签比例，可以估计分割结果质量，不写不进行统计",
)
args = parser.parse_args()


def get_name(name):
    """从文件名中解析扫描序列的名字和片层下标

    Parameters
    ----------
    name : str
        片层标签文件名

    Returns
    -------
    str, int
        序列名，用于group一个序列的所有推理结果
        序列下标，表示这个片层在序列中的下标

    """
    # TODO: rfind
    pos = -name[::-1].find("-") - 1  # 找到最后一个 -
    # print(name, name[:pos], int(name[pos + 1 :].split(".")[0]))
    return name[:pos], int(name[pos + 1 :].split(".")[0])


# TODO: 多进程不是多线程
class ThreadPool(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, maxsize=6, *args, **kwargs):
        super(ThreadPool, self).__init__(*args, **kwargs)
        self._work_queue = Queue(maxsize=maxsize)


# 检查文件匹配情况
img_names = util.listdir(args.png_dir, sort=False)
patient_names = []
for n in img_names:
    n, _ = get_name(n)
    if n not in patient_names:
        patient_names.append(n)
patient_names = [n + ".nii.gz" for n in patient_names]

nii_names_set = set(os.listdir(args.scan_dir))
patient_names_set = set(patient_names)
for n in patient_names_set - nii_names_set:
    print(n, "dont have nii")
for n in nii_names_set - patient_names_set:
    print(n, "dont have segmentation result")
patient_names.sort()
print(patient_names)

input("Press any key to start！")
if args.percent:
    percent_file = open(args.percent, "a+")
if not os.path.exists(args.seg_dir):
    os.makedirs(args.seg_dir)


def run(patient):
    if osp.exists(osp.join(args.seg_dir, patient)):
        print(patient, "already finished, skipping")
        return

    patient_imgs = [n for n in img_names if get_name(n)[0] == patient.split(".")[0]]
    patient_imgs.sort(key=lambda n: int(get_name(n)[1]))
    # print(patient, patient_imgs, len(patient_imgs))
    label = cv2.imread(
        os.path.join(args.png_dir, patient_imgs[0]), cv2.IMREAD_UNCHANGED
    )
    s = label.shape
    label_data = np.zeros([s[0], s[1], len(patient_imgs)], dtype="uint8")

    try:
        # print(os.path.join(args.scan_dir, patient))
        scanf = nib.load(os.path.join(args.scan_dir, patient))
        scan_header = scanf.header
    except:
        print(f"[ERROR] {patient}'s scan is not found! Skipping {patient}")
        return
        # scanf = nib.load(os.path.join(args.scan_dir, "张金华_20201024213424575a.nii"))
        # scan_header = scanf.header

    for img_name in patient_imgs:
        img = cv2.imread(os.path.join(args.png_dir, img_name), cv2.IMREAD_UNCHANGED)
        ind = int(get_name(img_name)[1])
        label_data[:, :, ind] = img

    save_nii(
        label_data,
        scanf.affine,
        scan_header,
        os.path.join(args.seg_dir, patient),
    )  # BUG: 貌似会出现最后一两个进程卡住，无法保存的情况


if args.percent:
    percent_file.close()


def save_nii(label_data, affine, header, dir):
    print("++++", dir)
    print(label_data.shape)
    label_data = np.rot90(label_data, args.rot, axes=(0, 1))
    label_data = np.transpose(label_data, [1, 0, 2])
    if args.filter:
        tot = label_data.sum()
        label_data = util.filter_largest_volume(label_data, mode="hard")
        largest = label_data.sum()
        if args.percent:
            print(osp.basename(dir), largest / tot, file=percent_file)
            percent_file.flush()
    newf = nib.Nifti1Image(label_data.astype(np.float64), affine, header)
    nib.save(newf, dir)
    print("--------", "finish", dir)


print(patient_names)
with Pool(multiprocessing.cpu_count()) as p:
    p.map(run, patient_names)
