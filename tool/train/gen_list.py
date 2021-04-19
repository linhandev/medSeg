# 按照文件名排序，生成文件列表
import os
import os.path as osp
import argparse
import logging

from tqdm import tqdm
import numpy as np

import util

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, help="数据集路径", default=None)
parser.add_argument(
    "--img_fdr",
    type=str,
    help="扫描文件夹",
    default="JPEGImages",
)
parser.add_argument(
    "--lab_fdr",
    type=str,
    help="标签文件路径",
    default="Annotations",
)
parser.add_argument("-d", "--delimiter", type=str, help="分隔符", default=" ")
parser.add_argument(
    "-s",
    "--split",
    nargs=3,
    help="训练/验证/测试划分比例，比如 7 2 1",
    default=["7", "2", "1"],
)
args = parser.parse_args()

imgs = util.listdir(osp.join(args.base_dir, args.img_fdr))
labs = util.listdir(osp.join(args.base_dir, args.lab_fdr))
assert len(imgs) == len(
    labs
), f"Scan slice number ({len(imgs)}) isn't equal to label slice number({len(labs)})"

names = [[i, l] for i, l in zip(imgs, labs)]
file_names = ["train_list.txt", "eval_list.txt", "test_list.txt"]
split = util.toint(args.split)
tot = np.sum(split)
split = [int(s / tot * len(names)) for s in split]
print(f"Train/Eval/Test split is {split}")
split[1] += split[0]
split[2] = tot
part = 0
f = open(osp.join(args.base_dir, file_names[part]), "w")
for idx, (img, lab) in enumerate(names):
    if idx == split[part] and idx != len(names) - 1:
        f.close()
        part += 1
        f = open(osp.join(args.base_dir, file_names[part]), "w")

    print(
        "{:s}{:s}{:s}".format(
            osp.join(args.img_fdr, img),
            args.delimiter,
            osp.join(args.lab_fdr, lab),
        ),
        file=f,
    )
