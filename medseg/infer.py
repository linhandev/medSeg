from __future__ import print_function

import sys
import argparse
import time
from tqdm import tqdm
import os

import numpy as np
import cv2
import nibabel as nib
import paddle
from paddle import fluid


import utils.util as util
from utils.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="预测")
    parser.add_argument("-c", "--cfg_file", type=str, help="配置文件路径")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="使用GPU推理")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    if args.use_gpu:  # 命令行参数只能从false改成true，不能声明false
        cfg.TRAIN.USE_GPU = True

    cfg.set_immutable(True)


def main():
    places = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
    exe = fluid.Executor(places)

    infer_exe = fluid.Executor(places)
    inference_scope = fluid.core.Scope()

    if not os.path.exists(cfg.INFER.PATH.OUTPUT):
        os.makedirs(cfg.INFER.PATH.OUTPUT)

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
            cfg.INFER.PATH.PARAM, infer_exe
        )

        inf_volumes = os.listdir(cfg.INFER.PATH.INPUT)
        for inf_volume in tqdm(inf_volumes, position=0):
            total_time = 0
            read_time = 0
            inf_time = 0
            write_time = 0
            post_time = 0

            total_time = time.time()

            inf_path = os.path.join(cfg.INFER.PATH.INPUT, inf_volume)

            read_time = time.time()
            volf = nib.load(inf_path)
            volume = np.array(volf.get_fdata())
            if cfg.INFER.WINDOWLIZE:
                volume = util.windowlize_image(volume, cfg.INFER.WWWC)
            if cfg.INFER.DO_INTERP:
                header = volf.header.structarr
                # pixdim 是这套 ct 三个维度的间距
                pixdim = [header["pixdim"][ind] for ind in range(1, 4)]
                spacing = list(cfg.INFER.SPACING)
                for ind in range(3):
                    if spacing[ind] == -1:
                        spacing[ind] = pixdim[ind]
                ratio = [pixdim[0] / spacing[0], pixdim[1] / spacing[1], pixdim[2] / spacing[2]]
                volume = scipy.ndimage.interpolation.zoom(volume, ratio, order=3)

            inference = np.zeros(volume.shape)
            read_time = time.time() - read_time

            batch_size = cfg.INFER.BATCH_SIZE
            ind = 0
            flag = True
            pbar = tqdm(total=volume.shape[2] - 2, position=1, leave=False)
            pbar.set_postfix(file=inf_volume)
            while flag:
                tic = time.time()
                batch_data = []
                for j in range(0, batch_size):
                    ind = ind + 1
                    pbar.update(1)
                    data = volume[:, :, ind - 1 : ind + 2]
                    data = (
                        data.swapaxes(0, 2)
                        .reshape([3, data.shape[1], data.shape[0]])
                        .astype("float32")
                    )
                    batch_data.append(data)

                    if ind == volume.shape[2] - 2:
                        flag = False
                        pbar.refresh()
                        break
                toc = time.time()
                batch_data = np.array(batch_data)
                read_time += toc - tic

                tic = time.time()
                result = infer_exe.run(
                    inference_program,
                    feed={feed_target_names[0]: batch_data},
                    fetch_list=fetch_targets,
                )
                toc = time.time()
                inf_time += toc - tic

                tic = time.time()
                result = np.array(result)
                result = result.reshape([-1, 2, 512, 512])

                ii = ind
                for j in range(result.shape[0] - 1, -1, -1):
                    resp = result[j, 1, :, :].reshape([512, 512])
                    resp = resp.swapaxes(0, 1)
                    inference[:, :, ii] = resp
                    ii = ii - 1

                toc = time.time()
                post_time += toc - tic
            print("here")
            inference[inference >= cfg.INFER.THRESH] = 1
            inference[inference < cfg.INFER.THRESH] = 0
            if cfg.INFER.FILTER_LARGES:
                inference = util.filter_largest_volume(inference)
            if cfg.INFER.DO_INTERP:
                ratio = [1 / x for x in ratio]
                inference = scipy.ndimage.interpolation.zoom(inference, ratio, order=3)

            inference = inference.astype("int8")
            assert inference.size == volume.size, "预测输出大小和输入数据大小不同"
            write_time = time.time()
            inference_file = nib.Nifti1Image(inference, volf.affine)
            inferece_path = os.path.join(cfg.INFER.PATH.OUTPUT, inf_volume).replace(
                "volume", "segmentation"
            )
            nib.save(inference_file, inferece_path)
            write_time = time.time() - write_time

            total_time = time.time() - total_time

            print("Total process time: " + str(total_time))
            print("\t read file time: " + str(read_time))
            print("\t inference time: " + str(inf_time))
            print("\t post process time: " + str(post_time))
            print("\t write file time: " + str(write_time))
            print("\n\n\n")

            pbar.close()


if __name__ == "__main__":
    parse_args()
    main()
