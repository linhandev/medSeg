"""
将nii和标签批量转成2D切片(png/npy)
要求扫描和标签按照字典序排序相同，文件名相同，拓展名不同就可以满足这个
"""

import os
import os.path as osp
import argparse
import logging
from tqdm import tqdm

import util

parser = argparse.ArgumentParser()
parser.add_argument("--scan_dir", type=str, help="扫描文件路径", required=True)
parser.add_argument("--label_dir", type=str, help="标签文件路径", default=None)
parser.add_argument("--out_dir", type=str, help="数据集输出路径", required=True)
parser.add_argument(
    "-t",
    "--thresh",
    type=int,
    help="前景数量超过这个数才包含到数据集里，否则这个slice跳过",
    default=None,
)
parser.add_argument(
    "-s",
    "--size",
    nargs=2,
    help="输出片的大小，不声明这个参数不进行任何插值，否则扫描3阶插值，标签1阶插值到这个大小",
    default=None,
)
parser.add_argument("--wwwc", nargs=2, help="窗宽窗位", default=["1000", "0"])
parser.add_argument("-r", "--rot", type=int, help="逆时针转90度多少次", default=0)  # TODO: 用库做体位校正
parser.add_argument("-f", "--front", type=int, help="如果标签有多种前景，要保留的前景值", default=None)
parser.add_argument(
    "-fm",
    "--front_mode",
    type=str,
    help="多个前景保留一个的策略。stack：把大于front的标签都设成front，小于front的标签设成背景。single：只保留front，其他的都设成背景",
    default=None,
)
parser.add_argument(
    "-itv",
    "--interval",
    type=int,
    help="每隔这个数量取1片，重建层间距很小，片层很多的时候可以用这个跳过一些片",
    default=1,
)
parser.add_argument("-c", "--check", type=bool, help="是否检查数据集", default=False)
parser.add_argument("--remove", type=bool, help="是否删除检查中没有对应标签的扫描", default=False)
args = parser.parse_args()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# TODO: 完善对扫描和标签的检查
if args.check:
    util.check_nii_match(args.scan_dir, args.label_dir, remove=args.remove)

logging.info("Discovered scan/label pairs:")
scans = util.listdir(args.scan_dir)
if args.label_dir is not None:
    labels = util.listdir(args.label_dir)
    for s, l in zip(scans, labels):
        logging.info(s + "\t" + l)

cmd = input(
    f"""Totally {len(scans)} pairs, plz check for any mismatch.
    Input Y/y to continue, input anything else to stop: """
)
if cmd.lower() != "y":
    exit("Exit on user command")

progress = tqdm(range(len(scans)))
# TODO: 研究resize

for scan, label in zip(scans, labels):
    progress.set_description(f"Processing {osp.basename(scan)}")
    util.slice_med(
        osp.join(args.scan_dir, scan),
        osp.join(args.out_dir, "JPEGImages"),
        osp.join(args.label_dir, label) if args.label_dir else None,
        osp.join(args.out_dir, "Annotations") if args.label_dir else None,
        rot=args.rot,
        wwwc=util.toint(args.wwwc),
        thresh=args.thresh,
        front=args.front,
        front_mode=args.front_mode,
        itv=args.interval,
    )
