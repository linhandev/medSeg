import sys
import argparse
import time
import os
from multiprocessing import Process, Queue

from tqdm import tqdm
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


def read_data(file_path, q):
    """读取数据并进行预处理.

    Parameters
    ----------
    file_path : str
        需要读取的数据文件名.
    q : queue
        读入的数据放入这个队列.

    """
    print("Start Reading: ", file_path)
    volf = nib.load(file_path)
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
    q.put([volume, volf.affine])
    print("Finish Reading: ", file_path)


def post_process(fpath, inference, affine):
    """这个函数对数据进行后处理和存盘，防止GPU空等.

    Parameters
    ----------
    fpath : str
        要保存的文件名称.
    inference : ndarray
        输出的分割标签数组.
    """
    print("Start Postprocess: ", fpath)
    inference[inference >= cfg.INFER.THRESH] = 1
    inference[inference < cfg.INFER.THRESH] = 0
    if cfg.INFER.FILTER_LARGES:
        inference = util.filter_largest_volume(inference, 1, "soft")
    if cfg.INFER.DO_INTERP:
        ratio = [1 / x for x in ratio]
        inference = scipy.ndimage.interpolation.zoom(inference, ratio, order=3)

    inference = inference.astype("int8")
    inference_file = nib.Nifti1Image(inference, affine)
    inferece_path = fpath
    nib.save(inference_file, inferece_path)
    print("Finish POstprocess: ", fpath)


def main():
    places = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
    exe = fluid.Executor(places)

    infer_exe = fluid.Executor(places)
    inference_scope = fluid.core.Scope()
    vol_queue = Queue()

    if not os.path.exists(cfg.INFER.PATH.OUTPUT):
        os.makedirs(cfg.INFER.PATH.OUTPUT)

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
            cfg.INFER.PATH.PARAM, infer_exe
        )

        inf_volumes = os.listdir(cfg.INFER.PATH.INPUT)
        for vol_ind in tqdm(range(len(inf_volumes)), position=0):
            inf_volume = inf_volumes[vol_ind]
            if vol_queue.empty():
                read_data(os.path.join(cfg.INFER.PATH.INPUT, inf_volumes[vol_ind]), vol_queue)

            volume, affine = vol_queue.get()
            print(volume.shape)

            # 这里异步调一个读取数据，必须在get后面，队列是有锁的
            if vol_ind != len(inf_volumes):
                p = Process(
                    target=read_data,
                    args=(os.path.join(cfg.INFER.PATH.INPUT, inf_volumes[vol_ind + 1]), vol_queue),
                )
                p.start()
                # read_data(inf_path, vol_queue)

            inference = np.zeros(volume.shape)

            batch_size = cfg.INFER.BATCH_SIZE
            ind = 0
            flag = True
            pbar = tqdm(total=volume.shape[2] - 2, position=1, leave=False)
            pbar.set_postfix(file=inf_volume)
            while flag:
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
                batch_data = np.array(batch_data)

                result = infer_exe.run(
                    inference_program, feed={feed_target_names[0]: batch_data}, fetch_list=fetch_targets,
                )

                result = np.array(result)
                result = result.reshape([-1, 2, 512, 512])

                ii = ind
                for j in range(result.shape[0] - 1, -1, -1):
                    resp = result[j, 1, :, :].reshape([512, 512])
                    resp = resp.swapaxes(0, 1)
                    inference[:, :, ii] = resp
                    ii = ii - 1

                # 这里调用多进程后处理和存储
            p = Process(
                target=post_process,
                args=(
                    os.path.join(cfg.INFER.PATH.OUTPUT, inf_volume).replace("volume", "segmentation"),
                    inference,
                    affine,
                ),
            )
            p.start()
            # post_process(
            #     os.path.join(cfg.INFER.PATH.OUTPUT, inf_volume).replace("volume", "segmentation"),
            #     inference,
            #     affine,
            # )
            pbar.close()


if __name__ == "__main__":
    parse_args()
    main()
