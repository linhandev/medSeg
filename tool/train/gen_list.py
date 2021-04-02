# 按照文件名排序，生成文件列表
import os
import os.path as osp
import argparse
import logging  # 引入logging模块
from tqdm import tqdm

import util

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, help="数据集基路径", default=None)
parser.add_argument("--img_fdr", type=str, help="扫描文件夹", required=True)
parser.add_argument("--lab_fdr", type=str, help="标签文件路径", required=True)
parser.add_argument("-d", "--delimiter", type=str, help="分隔符", default=" ")
args = parser.parse_args()

imgs = util.listdir(osp.join(args.base_dir, args.img_fdr))
labs = util.listdir(osp.join(args.base_dir, args.lab_fdr))
names = [[i, l] for i, l in zip(imgs, labs)]


with open(osp.join(args.base_dir, "train.txt"), "w") as f:
    for img, lab in names:
        print(
            "{:s}{:s}{:s}".format(
                osp.join(args.img_fdr, img),
                args.delimiter,
                osp.join(args.lab_fdr, lab),
            ),
            file=f,
        )
