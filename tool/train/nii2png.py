# 将两个文件夹的nii扫描和标签转换成
"""
1. 所有的nii和标签转成png
2. 按照paddleseg的目录结构移动文件
"""
import os
import os.path as osp
import argparse
import logging  # 引入logging模块
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
parser.add_argument("--ww", type=int, help="窗宽", default=1000)
parser.add_argument("--wc", type=int, help="窗位", default=0)
parser.add_argument("-r", "--rot", type=int, help="逆时针转90度多少次", default=0)  # TODO: 用库做体位校正
parser.add_argument("-f", "--front", type=int, help="要保留的前景值", default=None)
parser.add_argument("-fm", "--front_mode", type=str, help="", default=None)
parser.add_argument("-itv", "--interval", type=int, help="每隔这个数量取1片，用于重建层很薄的情况", default=1)
parser.add_argument("-c", "--check", type=bool, help="是否检查数据集", default=False)
parser.add_argument("--remove", type=bool, help="", default=False)
args = parser.parse_args()
# TODO: 添加层厚功能
# TODO: 添加npy格式

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)

# CHECK: 完善对扫描和标签的检查
if args.check:
    util.check_nii_match(args.scan_dir, args.label_dir, remove=args.remove)


scans = util.listdir(args.scan_dir)
if args.label_dir is not None:
    labels = util.listdir(args.label_dir)
    # both = list(set(scans).intersection(labels))
    # scans = both
    # labels = both
    for s, l in zip(scans, labels):
        logging.info(s + "\t" + l)

cmd = input(f"Totally {len(scans)} pairs, Input Y/y to continue: ")
if cmd.lower() != "y":
    exit("exit on user cmd")

scans = tqdm(scans)
# TODO: 两种情况合并，删掉这个if
# TODO: 研究resize
if args.label_dir is not None:
    for scan, label in zip(scans, labels):
        scans.set_description("Processing {}".format(scan.rstrip(".gz").rstrip(".nii")))
        util.nii2png(
            os.path.join(args.scan_dir, scan),
            osp.join(args.out_dir, "JPEGImages"),
            os.path.join(args.label_dir, label) if args.label_dir else None,
            osp.join(args.out_dir, "Annotations"),
            rot=args.rot,
            wwwc=[args.ww, args.wc],
            thresh=args.thresh,
            front=args.front,
            front_mode=args.front_mode,
            itv=args.interval,
        )
else:
    # TODO: 对第一片和最后一片复制ｎ层，不要直接跳过
    for scan in scans:
        scans.set_description("Processing {}".format(scan.rstrip(".gz").rstrip(".nii")))
        util.nii2png(
            os.path.join(args.scan_dir, scan),
            osp.join(args.out_dir, "JPEGImages"),
            rot=args.rot,
            wwwc=[args.ww, args.wc],
            thresh=args.thresh,
        )
