"""
扫描数据集，生成summary
"""
# TODO: 添加窗宽窗位
# TODO: check大小是否一样

import argparse
import os
import os.path as osp

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

import util

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--scan_dir",
    type=str,
    required=True,
    help="扫描路径",
)
parser.add_argument("-l", "--label_dir", type=str, help="标签路径")
parser.add_argument("-p", "--plt_dir", type=str, help="强度分布输出路径，不写不进行绘制", default=None)
parser.add_argument(
    "--wwwc",
    nargs=2,
    default=None,
    help="窗宽窗位，不写不进行窗宽窗位处理",
)
parser.add_argument(
    "--skip",
    nargs=2,
    default=["", ""],
    help="在检查文件配对的时候，扫描和标签文件开头略过两个字符串，用于匹配 scan-1.nii 和 label-1.nii 这种情况",
)
args = parser.parse_args()

font = FontProperties(fname="../SimHei.ttf", size=16)

util.check_nii_match(args.scan_dir, args.label_dir, args.skip)

scans = util.listdir(args.scan_dir)
labels = util.listdir(args.label_dir)
assert len(scans) == len(labels), "扫描和标签数量不相等"

print(f"数据集中共{len(scans)}组扫描和标签，对应情况：")
for idx in range(len(scans)):
    print(f"{scans[idx]} \t {labels[idx]}")

if not osp.exists(osp.join(args.plt_dir, "scan")):
    os.makedirs(osp.join(args.plt_dir, "scan"))
if not osp.exists(osp.join(args.plt_dir, "label")):
    os.makedirs(osp.join(args.plt_dir, "label"))

pixdims = []
shapes = []
norms = []

pbar = tqdm(range(len(scans)), desc="正在统计")
for idx in range(len(scans)):
    pbar.set_postfix(filename=scans[idx].split(".")[0])
    pbar.update(1)

    scanf = nib.load(os.path.join(args.scan_dir, scans[idx]))
    labelf = nib.load(os.path.join(args.label_dir, labels[idx]))

    header = scanf.header.structarr
    shape = scanf.header.get_data_shape()
    shapes.append([shape[0], shape[1], shape[2]])
    pixdims.append(header["pixdim"][1:4])

    scan = scanf.get_fdata()
    norms.append([scan.min(), np.median(scan), scan.max()])

    if args.plt_dir:
        scan_plt = osp.join(args.plt_dir, "scan")
        label_plt = osp.join(args.plt_dir, "label")

        scan = scan.reshape([scan.size])

        plt.title(scans[idx].split(".")[0], fontproperties=font)
        plt.xlabel(
            "size:[{},{},{}] pixdims:[{},{},{}] ".format(
                shape[0],
                shape[1],
                shape[2],
                header["pixdim"][1],
                header["pixdim"][2],
                header["pixdim"][3],
            )
        )
        nums, bins, patchs = plt.hist(scan, bins=1000)
        plt.savefig(osp.join(scan_plt, scans[idx].split(".")[0] + ".png"))
        plt.close()

        file = open(osp.join(scan_plt, f"{scans[idx].split('.')[0]}.txt"), "w")
        print("--------- {} --------".format(scans[idx]), file=file)

        sum = 0
        for num in nums:
            sum += num
        nowsum = 0
        for i in range(0, len(nums)):
            nowsum += nums[i]
            print(
                "[{:<10f},{:<10f}] : {:>10} percentage : {}".format(
                    bins[i], bins[i + 1], nums[i], nowsum / sum
                ),
                file=file,
            )
        file.close()

        label = labelf.get_fdata()
        label = np.reshape(label, [label.size])
        plt.title(
            f"{scans[idx].split('.')[0]} [{np.min(label)},{np.max(label)}]",
            fontproperties=font,
        )
        plt.xlabel(
            "size:[{},{},{}] pixdims:[{},{},{}] ".format(
                shape[0],
                shape[1],
                shape[2],
                header["pixdim"][1],
                header["pixdim"][2],
                header["pixdim"][3],
            )
        )
        nums, bins, patchs = plt.hist(label, bins=5)
        plt.savefig(osp.join(label_plt, scans[idx].rstrip(".nii") + ".png"))
        plt.close()

        file = open(
            os.path.join(label_plt, "{}.txt".format(scans[idx].split(".")[0])),
            "w",
        )
        print("--------- {} --------".format(scans[idx]), file=file)

        sum = 0
        for num in nums:
            sum += num
        nowsum = 0
        for i in range(0, len(nums)):
            nowsum += nums[i]
            print(
                "[{:<10f},{:<10f}] : {:>10} percentage : {}".format(
                    bins[i], bins[i + 1], nums[i], nowsum / sum
                ),
                file=file,
            )

pbar.close()

spacing = np.median(pixdims, axis=0)
size = np.median(shapes, axis=0)

print(norms)
norm = []
norms = np.array(norms)
norm.append(np.min(norms[:, 0]))
norm.append(np.median(norms[:, 1]))
norm.append(np.max(norms[:, 2]))

print(spacing, size, norm)

file = open("./summary.txt", "w")

print("spacing", file=file)
for dat in spacing:
    print(dat, file=file)

print("\nsize", file=file)
for dat in size:
    print(dat, file=file)

print("\nnorm", file=file)
for dat in norm:
    print(dat, file=file)
file.close()
