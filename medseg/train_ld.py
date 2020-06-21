from __future__ import print_function

import sys
import os
import argparse
import random
import math
import numpy as np
from datetime import datetime

from tqdm.auto import tqdm


import paddle
import paddle.fluid as fluid
from paddle.utils.plot import Ploter
from paddle.fluid.layers import log

from visualdl import LogWriter

from models.unet_simple import unet_simple
from models.deeplabv3p import deeplabv3p
from models.unet_base import unet_base
from utils.util import *


import loss
import aug
from utils.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="训练")
    parser.add_argument("-c", "--cfg_file", type=str, help="配置文件路径")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="是否用GPU")
    parser.add_argument("--do_eval", action="store_true", default=False, help="是否进行测试")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    if args.use_gpu:  # 命令行参数只能从false改成true，不能声明false
        cfg.TRAIN.USE_GPU = True
    if args.do_eval:
        cfg.TRAIN.DO_EVAL = True

    cfg.set_immutable(True)


def data_reader(part_start=0, part_end=8, is_test=False):
    npz_names = listdir(cfg.TRAIN.DATA_PATH)
    npz_part = npz_names[len(npz_names) * part_start // 10 : len(npz_names) * part_end // 10]
    random.shuffle(npz_part)

    def reader():
        for npz_name in npz_part:
            data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz_name))
            vols = data["vols"]
            labs = data["labs"]
            for ind in range(vols.shape[0]):
                vol = vols[ind].reshape(3, 512, 512).astype("float32")
                lab = labs[ind].reshape(1, 512, 512).astype("int32")

                yield vol, lab

    return reader


def aug_mapper(data):
    vol = data[0]
    lab = data[1]
    return vol, lab


def main():

    train_program = fluid.Program()
    train_init = fluid.Program()

    with fluid.program_guard(train_program, train_init):
        volume = fluid.layers.data(name="volume", shape=[3, 512, 512], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1, 512, 512], dtype="int32")
        train_loader = fluid.io.DataLoader.from_generator(
            feed_list=[volume, label], capacity=16, iterable=False, use_double_buffer=True
        )
        prediction = unet_base(volume, 2, [512, 512])
        avg_loss = loss.create_loss(prediction, label, 2)
        miou = loss.mean_iou(prediction, label, 2)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.003)
        optimizer.minimize(avg_loss)

    places = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
    exe = fluid.Executor(places)
    exe.run(train_init)
    train_reader = fluid.io.xmap_readers(aug_mapper, data_reader(0, 8), 8, 32, False)
    # train_reader = data_reader(0, 8)
    train_loader.set_sample_generator(train_reader, batch_size=16, places=places)
    train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)

    step = 0
    best_miou = 0
    for pass_id in range(20):
        train_loader.start()
        try:
            loss_val, iou = exe.run(program=train_program, fetch_list=[avg_loss, miou])
            print("Train Step {} : Loss {} , MIOU {} .".format(step, loss_val, iou))
            step += 1
        except fluid.core.EOFException:
            print("end of epoch")
            train_loader.reset()

    # for data in train_loader():
    #     loss_val, iou = exe.run(program=train_program, feed=data, fetch_list=[avg_loss, miou])
    #     if step % 10 == 0:
    #         print("Train Step {} : Loss {} , MIOU {} .".format(step, loss_val, iou))
    #     step += 1


if __name__ == "__main__":
    args = parse_args()
    main()
