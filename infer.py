from __future__ import print_function

import sys
import argparse

import numpy
import cv2

import paddle
from paddle import fluid
from util import *
import nibabel as nib
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser("liverseg")
    parser.add_argument(
                        '--use_gpu',
                        type=bool,
                        default=True,
                        help="Whether to use GPU or not.")
    args = parser.parse_args()
    return args

def main():
    use_cuda = args.use_gpu
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer_param_path="/home/aistudio/work/params/tumor_unet_.77/inf/"
    infer_param_path="/home/aistudio/work/params/unet_base_16_iou965/inf/"
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
            # print(inf_path)

            read_time = time.time()
            volf = nib.load(inf_path)
            volume = np.array(volf.get_fdata())
            volume = volume.clip(-1024, 1024)
            inference = np.zeros(volume.shape)
            read_time = time.time() - read_time

            batch_size = 8
            ind = 0
            flag = True
            pbar = tqdm(total = volume.shape[2] - 2, position=1, leave=False)
            pbar.set_postfix(file=inf_volume)
            while flag:
                tic = time.time()
                batch_data = []
                for j in range(0,batch_size):
                    ind = ind + 1
                    pbar.update(1)
                    data = volume[:,:,ind - 1: ind + 2]
                    data = data.swapaxes(0,2).reshape([3,512,512]).astype('float32')
                    batch_data.append(data)

                    if ind == volume.shape[2] - 2:
                        flag = False
                        pbar.refresh()
                        break
                toc = time.time()
                batch_data = np.array(batch_data)
                read_time += toc - tic

                tic = time.time()
                result = infer_exe.run(inference_program, feed={feed_target_names[0]: batch_data }, fetch_list=fetch_targets)
                toc = time.time()
                inf_time += toc - tic

                tic = time.time()
                result = np.array(result)
                result = result.reshape([-1, 2, 512, 512])
                result[result > 0.9] = 2
                result[result < 0.9] = 0

                ii = ind
                for j in range(result.shape[0] - 1, -1, -1):
                    resp  = result[j, 1, :, :].reshape([512, 512])
                    resp = resp.swapaxes(0,1)
                    inference[:, :, ii] = resp
                    ii = ii - 1
                # result = result * 255
                # cv2.imwrite("result.png",result)

                toc = time.time()
                post_time += toc - tic

            inference = filter_largest_volume(inference)
            inference = inference.astype("int16")
            write_time = time.time()
            inference_file = nib.Nifti1Image(inference, volf.affine)
            nib.save(inference_file, os.path.join(inference_label_path, inf_volume))
            write_time = time.time() - write_time

            total_time = time.time() - total_time


            # print("Total process time: " + str(total_time) )
            # print("\t read file time: " + str(read_time))
            # print("\t inference time: " + str(inf_time))
            # print("\t post process time: " + str(post_time))
            # print("\t write file time: " + str(write_time))
            # print("\n\n\n")


            pbar.close()

if __name__ == '__main__':
    args = parse_args()
    main()
