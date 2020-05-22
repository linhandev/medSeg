# encoding=utf-8
import numpy as np
import nibabel as nib
from tqdm import tqdm
import scipy
from util import *
from config import *
from lib.threshold_function_module import windowlize_image
import argparse


"""
测试预处理代码，包含脚手架代码，保存成nii文件
写降噪和增强的代码
"""


parser = argparse.ArgumentParser(description="数据预处理")
parser.add_argument("--front", type=int, default=1, help="处理的目标前景")
parser.add_argument("--crop", action="store_true", default=False, help="是否切到只有前景")
parser.add_argument("--interp", action="store_true", default=False, help="是否进行插值")
parser.add_argument("--window", action="store_true", default=False, help="是否进行窗口化")
parser.add_argument("--plane", type=str, default="xy", help="处理的目标前景")
parser.add_argument("--thresh", type=int, default=512, help="只取前景数量超过thersh的slice")
args = parser.parse_args()
print(args)

if args.plane == "xy" and not os.path.exists(preprocess_path):
    os.makedirs(preprocess_path)
if args.plane == "xz" and not os.path.exists(z_prep_path):
    os.makedirs(z_prep_path)

volumes = listdir(volumes_path)
labels = listdir(labels_path)

pbar = tqdm(range(len(labels)), desc="数据处理中")
for i in range(len(labels)):

    pbar.set_postfix(filename=labels[i].rstrip(".nii"))
    pbar.update(1)

    assert volumes[i].lstrip("volume").rstrip(".gz") == labels[i].lstrip("segmentation").rstrip(".gz"), "文件名不匹配"

    volf = nib.load(os.path.join(volumes_path, volumes[i]))
    labf = nib.load(os.path.join(labels_path, labels[i]))

    save_info(volumes[i], volf.header, "./lits.csv")

    volume = volf.get_fdata()
    label = labf.get_fdata()

    if args.interp:
        header = volf.header.structarr
        spacing = [1, 1, 1]
        pixdim = [header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]]  # pixdim 是这张 ct 三个维度的间距
        ratio = [pixdim[0] / spacing[0], pixdim[1] / spacing[1], pixdim[2] / spacing[2]]
        ratio = [1, 1, ratio[2]]
        volume = scipy.ndimage.interpolation.zoom(volume, ratio, order=3)
        label = scipy.ndimage.interpolation.zoom(label, ratio, order=0)

    # volume=np.clip(volume,-1024,1024)
    if args.window:
        volume = windowlize_image(volume, 500, 30)  # ww wc

    label = clip_label(label, args.front)

    # if label.sum() < 32:
    #     continue

    if args.crop:
        bb_min, bb_max = get_bbs(label)
        label = crop_to_bbs(label, bb_min, bb_max, 0.5)[0]
        volume = crop_to_bbs(volume, bb_min, bb_max)[0]

        label = pad_volume(label, [512, 512, 0], 0)  # NOTE: 注意标签使用 0
        volume = pad_volume(volume, [512, 512, 0], -1024)
        print("after padding", volume.shape, label.shape)

    volume = volume.astype(np.float16)
    label = label.astype(np.int8)

    if args.plane == "xy":
        for frame in range(1, volume.shape[2] - 1):
            if np.sum(label[:, :, frame]) > args.thresh:
                vol = volume[:, :, frame - 1 : frame + 2]
                lab = label[:, :, frame]
                lab = lab.reshape([lab.shape[0], lab.shape[1], 1])

                vol = np.swapaxes(vol, 0, 2)
                lab = np.swapaxes(lab, 0, 2)  # [3,512,512],3 2 1 的顺序，用的时候倒回来, CWH

                data = np.concatenate((vol, lab), axis=0)
                # print(data.dtype)
                file_name = "lits_{}_f{}-{}-{}.npy".format(
                    args.plane, args.front, volumes[i].rstrip(".nii").lstrip("volume-"), frame
                )
                file_path = os.path.join(preprocess_path, file_name)
                np.save(file_path, data)
    else:
        if volume.shape[2] > 512:
            volume = volume[:, :, 0:512]
            label = label[:, :, 0:512]
        else:
            label = pad_volume(label, [0, 512, 512], 0)  # NOTE: 注意标签使用 0
            volume = pad_volume(volume, [0, 512, 512], -1024)

        for frame in range(1, volume.shape[0] - 1):  # 解决数据不足 512 的问题
            if np.sum(label[frame, :, :]) > args.thresh:
                vol = volume[frame - 1 : frame + 2, :, :]
                lab = label[frame, :, :]
                lab = lab.reshape([1, lab.shape[0], lab.shape[1]])

                print(vol.shape)
                print(lab.shape)

                assert vol.shape == (3, 512, 512), "vol shape incorrect, being{}".format(vol.shape)
                assert lab.shape == (1, 512, 512), "lab shape incorrect, being{}".format(lab.shape)

                data = np.concatenate((vol, lab), axis=0)
                file_name = "lits_{}_f{}-{}-{}.npy".format(
                    args.plane, args.front, volumes[i].rstrip(".nii").lstrip("volume-"), frame
                )
                np.save(os.path.join(z_prep_path, file_name), data)
pbar.close()
