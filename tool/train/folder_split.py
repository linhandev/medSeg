# 原来的数据和标签分别在一个目录里，进行随机split之后按照pdseg的目录结构放
import os
import argparse
import random

import util

# shutil.move("source", "destination")  # 递归移动

parser = argparse.ArgumentParser()
parser.add_argument("--dst_dir", type=str, required=True)
parser.add_argument("--img_folder", type=str, required=True)
parser.add_argument("--lab_folder", type=str, required=True)
args = parser.parse_args()


def mv(curr, dest):
    print(os.path.join(dest.rstrip(dest.split("/")[-1])))
    if not os.path.exists(os.path.join(dest.rstrip(dest.split("/")[-1]))):
        os.makedirs(os.path.join(dest.rstrip(dest.split("/")[-1])))
    os.rename(curr, dest)


def move(
    img_folder,
    lab_folder,
    dst_dir,
    split=[8, 2, 0],
    folders=["imgs", "annotations"],
    sub_folders=["train", "val", "test"],
):
    # 1. 创建目标目录
    for fd1 in folders:
        for fd2 in sub_folders:
            dir = os.path.join(dst_dir, fd1, fd2)
            if not os.path.exists(dir):
                os.makedirs(dir)

    # 2. 获取图像和标签文件名，打乱
    img_names = util.listdir(img_folder)
    lab_names = util.listdir(lab_folder)
    names = [[i, l] for i, l in zip(img_names, lab_names)]
    random.shuffle(names)

    for idx in range(10):
        print(names[idx])

    # 3. 计算划分点
    split.insert(0, 0)
    for ind in range(1, len(split)):
        split[ind] += split[ind - 1]

    split = [x / split[-1] for x in split]
    split = [int(len(img_names) * split[ind]) for ind in range(4)]
    print(split)

    # 4. 进行移动
    for part in range(3):
        print(f"正在处理{sub_folders[part]}")
        for idx in range(split[part], split[part + 1]):
            img, lab = names[idx]
            mv(
                os.path.join(img_folder, img),
                os.path.join(dst_dir, folders[0], sub_folders[part], img),
            )
            mv(
                os.path.join(lab_folder, lab),
                os.path.join(dst_dir, folders[1], sub_folders[part], lab),
            )


if __name__ == "__main__":
    move(
        img_folder=args.img_folder,
        lab_folder=args.lab_folder,
        dst_dir=args.dst_dir,
    )
