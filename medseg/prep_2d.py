# 将图片组batch，存成和3D预处理一样的npz格式
import os

import cv2
import argparse
import numpy as np

import utils.util as util
from utils.config import cfg
import aug


def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("-c", "--cfg_file", type=str, help="配置文件路径")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)


def main():
    images = util.listdir(cfg.DATA.INPUTS_PATH)
    labels = util.listdir(cfg.DATA.LABELS_PATH)
    print(images)

    if not os.path.exists(cfg.DATA.PREP_PATH):
        os.makedirs(cfg.DATA.PREP_PATH)
    npz_count = 0
    img_npz = []
    lab_npz = []
    for ind in range(len(images)):
        img = cv2.imread(os.path.join(cfg.DATA.INPUTS_PATH, images[ind]))
        lab = cv2.imread(os.path.join(cfg.DATA.LABELS_PATH, labels[ind]))
        img = img.swapaxes(0, 2)
        lab = lab.swapaxes(0, 2)

        lab = lab[0] / 255
        lab = lab[np.newaxis, :, :]

        img, lab = aug.crop(img, lab, size=[3, 512, 512])

        img_npz.append(img)
        lab_npz.append(lab)

        print(img.shape, lab.shape)

        if len(img_npz) == cfg.PREP.BATCH_SIZE or ind == len(images) - 1:
            imgs = np.array(img_npz)
            labs = np.array(lab_npz)
            file_name = "{}-{}".format(cfg.DATA.NAME, npz_count)
            file_path = os.path.join(cfg.DATA.PREP_PATH, file_name)
            np.savez(file_path, imgs=imgs, labs=labs)
            img_npz = []
            lab_npz = []
            npz_count += 1


if __name__ == "__main__":
    parse_args()
    main()
