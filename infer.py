from __future__ import print_function

import sys
import argparse

import numpy
import cv2

import paddle
from paddle import fluid
import nibabel as nib
from tqdm import tqdm
import time
from util import *
from config import *


def parse_args():
    parser = argparse.ArgumentParser("liverseg")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="用GPU推理")
    parser.add_argument("--batch_size", type=int, default=16, help="推理过程中的batch size")
    parser.add_argument("--type", type=str, default="liver", help="针对一个type有一套pipline和权重路径")
    parser.add_argument("--interp", type="store_true", default=False, help="是否对数据进行插值，用于z方向网络")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    use_cuda = args.use_gpu
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    if args.type == "liver":
        infer_param_path = "/home/aistudio/weights/liver/inf"
    elif args.type == "tumor":
        infer_param_path = "/home/aistudio/weights/tumor/inf"
    else:
        raise Exception("错误的前景类别")

    if not os.path.exists(inference_label_path):
        os.makedirs(inference_label_path)

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(infer_param_path, infer_exe)

        inf_volumes = os.listdir(inference_path)
        for inf_volume in tqdm(inf_volumes, position=0):
            total_time = 0
            read_time = 0
            inf_time = 0
            write_time = 0
            post_time = 0

            total_time = time.time()

            inf_path = os.path.join(inference_path, inf_volume)

            read_time = time.time()
            volf = nib.load(inf_path)
            volume = np.array(volf.get_fdata())
            volume = windowlize_image(volume, 200, 70)
            if args.interp:
                header = volf.header.structarr
                spacing = [1, 1, 1]
                pixdim = [header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]]  # pixdim 是这张 ct 三个维度的间距
                ratio = [pixdim[0] / spacing[0], pixdim[1] / spacing[1], pixdim[2] / spacing[2]]
                ratio = [1, 1, ratio[2]]
                volume = scipy.ndimage.interpolation.zoom(volume, ratio, order=3)

            inference = np.zeros(volume.shape)
            read_time = time.time() - read_time

            batch_size = args.batch_size
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
                    data = data.swapaxes(0, 2).reshape([3, data.shape[1], data.shape[0]]).astype("float32")
                    batch_data.append(data)

                    if ind == volume.shape[2] - 2:
                        flag = False
                        pbar.refresh()
                        break
                toc = time.time()
                batch_data = np.array(batch_data)
                read_time += toc - tic

                tic = time.time()
                result = infer_exe.run(inference_program, feed={feed_target_names[0]: batch_data}, fetch_list=fetch_targets)
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
                # result = result * 255
                # cv2.imwrite("result.png",result)

                toc = time.time()
                post_time += toc - tic

            # inference = filter_largest_volume(inference)
            # inference[30:60, 30:60, 30:60] = 2
            if args.interp:
                ratio = [1 / x for x in ratio]
                inference = scipy.ndimage.interpolation.zoom(inference, ratio, order=3)

            inference[inference >= 0.9] = 1
            inference[inference < 0.9] = 0
            if args.type == "tumor":
                inference = inference * 2

            inference = inference.astype("int8")
            assert inference.size == volume.size, "预测输出大小和输入数据大小不同"
            write_time = time.time()
            inference_file = nib.Nifti1Image(inference, volf.affine)
            inferece_path = os.path.join(inference_label_path, inf_volume).replace("volume", "segmentation")
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
    args = parse_args()
    main()
