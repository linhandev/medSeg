# encoding=utf-8


import os


import numpy as np
import nibabel as nib
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt

import utils.util as util
from utils.config import cfg
import utils.util as util

import argparse
import aug

# import vis

np.set_printoptions(threshold=np.inf)


"""
对 3D 体数据进行一些预处理，并保存成npz文件
每个npz文件包含volume和label两个数组，volume和label各包含n条扫描记录，文件进行压缩
"""
# TODO: 支持更多的影像格式
# TODO: 提供预处理npz gzip选项
# https://stackoverflow.com/questions/54238670/what-is-the-advantage-of-saving-npz-files-instead-of-npy-in-python-regard


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
    # 1. 创建输出路径，删除非空的summary表格
    if cfg.PREP.PLANE == "xy" and not os.path.exists(cfg.DATA.PREP_PATH):
        os.makedirs(cfg.DATA.PREP_PATH)
    if cfg.PREP.PLANE == "xz" and not os.path.exists(cfg.DATA.Z_PREP_PATH):
        os.makedirs(cfg.DATA.Z_PREP_PATH)

    if os.path.exists(cfg.DATA.SUMMARY_FILE) and os.path.getsize(cfg.DATA.SUMMARY_FILE) != 0:
        os.remove(cfg.DATA.SUMMARY_FILE)

    volumes = util.listdir(cfg.DATA.INPUTS_PATH)
    labels = util.listdir(cfg.DATA.LABELS_PATH)

    # 获取体位信息
    f = open("./directions.csv")
    dirs = f.readlines()
    # print("dirs: ", dirs)
    dirs = [x.rstrip("\n") for x in dirs]
    dirs = [x.split(",") for x in dirs]
    dic = {}
    for dir in dirs:
        dic[dir[0].strip()] = dir[1].strip()
    f.close()

    vol_npz = []
    lab_npz = []
    npz_count = 0
    thick = (cfg.TRAIN.THICKNESS - 1) / 2
    pbar = tqdm(range(len(labels)), desc="数据处理中")
    for i in range(len(labels)):
        pbar.set_postfix(filename=labels[i] + " " + volumes[i])
        pbar.update(1)

        print(volumes[i], labels[i])

        volf = nib.load(os.path.join(cfg.DATA.INPUTS_PATH, volumes[i]))
        labf = nib.load(os.path.join(cfg.DATA.LABELS_PATH, labels[i]))

        util.save_info(volumes[i], volf.header, cfg.DATA.SUMMARY_FILE)

        volume = volf.get_fdata()
        label = labf.get_fdata()
        label = label.astype(int)
        # plt.imshow(volume[:, :, 0])
        # plt.show()
        if dic[volumes[i]] == "2":
            volume = np.rot90(volume, 3)
            label = np.rot90(label, 3)
        else:
            volume = np.rot90(volume, 1)
            label = np.rot90(label, 1)

        # plt.imshow(volume[:, :, 0])
        # plt.show()

        if cfg.PREP.INTERP:
            print("interping")
            header = volf.header.structarr
            spacing = cfg.PREP.INTERP_PIXDIM
            # pixdim 是 ct 三个维度的间距
            pixdim = [header["pixdim"][x] for x in range(1, 4)]
            for ind in range(3):
                if spacing[ind] == -1:  # 如果目标spacing为 -1 ,这个维度不进行插值
                    spacing[ind] = pixdim[ind]
            ratio = [x / y for x, y in zip(spacing, pixdim)]
            volume = scipy.ndimage.interpolation.zoom(volume, ratio, order=3)
            label = scipy.ndimage.interpolation.zoom(label, ratio, order=0)

        if cfg.PREP.WINDOW:
            volume = util.windowlize_image(volume, cfg.PREP.WWWC)

        label = util.clip_label(label, cfg.PREP.FRONT)

        if cfg.PREP.CROP:  # 裁到只有前景
            bb_min, bb_max = get_bbs(label)
            label = crop_to_bbs(label, bb_min, bb_max, 0.5)[0]
            volume = crop_to_bbs(volume, bb_min, bb_max)[0]

            label = pad_volume(label, [512, 512, 0], 0)  # NOTE: 注意标签使用 0
            volume = pad_volume(volume, [512, 512, 0], -1024)
            print("after padding", volume.shape, label.shape)

        volume = volume.astype(np.float16)
        label = label.astype(np.int8)

        crop_size = list(cfg.PREP.SIZE)
        for ind in range(3):
            if crop_size[ind] == -1:
                crop_size[ind] = volume.shape[ind]
        volume, label = aug.crop(volume, label, crop_size)

        # 开始切片
        if cfg.PREP.PLANE == "xy":
            for frame in range(1, volume.shape[2] - 1):
                if label[:, :, frame].sum() > cfg.PREP.THRESH:
                    vol = volume[:, :, frame - thick : frame + thick + 1]
                    lab = label[:, :, frame]
                    lab = lab[:, :, np.newaxis]

                    vol = np.swapaxes(vol, 0, 2)
                    lab = np.swapaxes(lab, 0, 2)  # [3,512,512],CWH 的顺序

                    vol_npz.append(vol.copy())
                    lab_npz.append(lab.copy())
                    print("{} 片满足，当前共 {}".format(frame, len(vol_npz)))

                    if len(vol_npz) == cfg.PREP.BATCH_SIZE or (
                        i == (len(labels) - 1) and frame == volume.shape[2] - 1
                    ):
                        imgs = np.array(vol_npz)
                        labs = np.array(lab_npz)
                        print(imgs.shape)
                        print(labs.shape)
                        print("正在存盘")
                        file_name = "{}_{}_f{}-{}".format(
                            cfg.DATA.NAME, cfg.PREP.PLANE, cfg.PREP.FRONT, npz_count
                        )
                        file_path = os.path.join(cfg.DATA.PREP_PATH, file_name)
                        np.savez(file_path, imgs=imgs, labs=labs)
                        vol_npz = []
                        lab_npz = []
                        npz_count += 1
        else:
            print(volume.shape, label.shape)
            for frame in range(1, volume.shape[0] - 1):
                if label[frame, :, :].sum() > cfg.PREP.THRESH:
                    vol = volume[frame - 1 : frame + 2, :, :]
                    lab = label[frame, :, :]
                    lab = lab.reshape([1, lab.shape[0], lab.shape[1]])

                    vol_npz.append(vol.copy())
                    lab_npz.append(lab.copy())

                    if len(vol_npz) == cfg.PREP.BATCH_SIZE:
                        vols = np.array(vol_npz)
                        labs = np.array(lab_npz)
                        print(vols.shape)
                        print(labs.shape)
                        print("正在存盘")
                        file_name = "{}_{}_f{}-{}".format(
                            cfg.DATA.NAME, cfg.PREP.PLANE, cfg.PREP.FRONT, npz_count
                        )
                        file_path = os.path.join(cfg.DATA.Z_PREP_PATH, file_name)
                        np.savez(file_path, vols=vols, labs=labs)
                        vol_npz = []
                        lab_npz = []
                        npz_count += 1

    pbar.close()


if __name__ == "__main__":
    parse_args()
    main()
