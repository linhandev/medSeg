# coding=utf-8
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
from paddle.fluid.layers import log
from visualdl import LogWriter

import utils.util as util
from utils.config import cfg
from models.unet_base import unet_base
from models.unet_simple import unet_simple
from models.deeplabv3p import deeplabv3p
import loss
import aug


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
    npz_names = util.listdir(cfg.TRAIN.DATA_PATH)
    npz_part = npz_names[len(npz_names) * part_start // 10 : len(npz_names) * part_end // 10]
    random.shuffle(npz_part)

    def reader():
        # BUG: tqdm每次更新都另起一行，此外要单独测试windows上好不好使
        if cfg.TRAIN.DATA_COUNT != -1:
            pbar = tqdm(total=cfg.TRAIN.DATA_COUNT, desc="训练进度")
        for npz_name in npz_part:
            data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz_name))
            vols = data["vols"]
            labs = data["labs"]
            for ind in range(vols.shape[0]):
                if cfg.TRAIN.DATA_COUNT != -1:
                    pbar.update()
                vol = vols[ind].reshape(3, 512, 512).astype("float32")
                lab = labs[ind].reshape(1, 512, 512).astype("int32")
                if cfg.AUG.WINDOWLIZE:
                    vol = util.windowlize_image(vol, cfg.AUG.WWWC)  # 肝脏常用
                yield vol, lab

    return reader


def aug_mapper(data):
    vol = data[0]
    lab = data[1]
    # NOTE: 注意不要增强第0维,那是厚度的方向
    vol, lab = aug.flip(vol, lab, cfg.AUG.FLIP.RATIO)
    ww, wc = cfg.AUG.WWWC
    vol, lab = aug.rotate(vol, lab, cfg.AUG.ROTATE.RANGE, cfg.AUG.ROTATE.RATIO, wc - ww / 2)
    vol, lab = aug.zoom(vol, lab, cfg.AUG.ZOOM.RANGE, cfg.AUG.ZOOM.RATIO)
    vol, lab = aug.crop(vol, lab, cfg.AUG.CROP.SIZE)
    return vol, lab


def main():

    train_program = fluid.Program()
    train_init = fluid.Program()

    with fluid.program_guard(train_program, train_init):
        volume = fluid.layers.data(name="volume", shape=[3, 512, 512], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1, 512, 512], dtype="int32")
        train_loader = fluid.io.DataLoader.from_generator(
            feed_list=[volume, label],
            capacity=cfg.TRAIN.BATCH_SIZE * 2,
            iterable=True,
            use_double_buffer=True,
        )
        # TODO: 用list实现
        if cfg.TRAIN.ARCHITECTURE == "unet_simple":
            prediction = unet_simple(volume, 2, [512, 512])
        elif cfg.TRAIN.ARCHITECTURE == "unet_base":
            prediction = unet_base(volume, 2, [512, 512])
        elif cfg.TRAIN.ARCHITECTURE == "deeplabv3":
            prediction = deeplabv3p(volume, 2)
        else:
            print("错误的网络类型")
            sys.exit(0)

        avg_loss = loss.create_loss(prediction, label, 2)
        miou = loss.mean_iou(prediction, label, 2)

        # TODO: l1 decay
        # decay = paddle.fluid.regularizer.L2Decay(0.0001)

        # TODO: sgd momentum
        # optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.003, regularization=decay)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.003)
        # optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.006, momentum=0.8,regularization=decay)

        optimizer.minimize(avg_loss)

    places = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
    exe = fluid.Executor(places)
    exe.run(train_init)
    exe_test = fluid.Executor(places)

    test_program = train_program.clone(for_test=True)
    train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)

    # BUG:
    # if cfg.TRAIN.PRETRAINED_WEIGHT:
    #     fluid.io.load_persistables(exe, cfg.TRAIN.PRETRAINED_WEIGHT, train_init)

    train_reader = fluid.io.xmap_readers(aug_mapper, data_reader(0, 8), 8, cfg.TRAIN.BATCH_SIZE * 2)
    train_loader.set_sample_generator(train_reader, batch_size=cfg.TRAIN.BATCH_SIZE, places=places)
    test_reader = paddle.batch(data_reader(8, 10, True), cfg.TRAIN.BATCH_SIZE)
    test_feeder = fluid.DataFeeder(place=places, feed_list=[volume, label])

    writer = LogWriter(logdir="/home/aistudio/log/{}".format(datetime.now()))

    step = 0
    best_miou = 0

    for pass_id in range(cfg.TRAIN.EPOCHS):
        for train_data in train_loader():
            step += 1
            avg_loss_value, miou_value = exe.run(
                train_program, feed=train_data, fetch_list=[avg_loss, miou]
            )
            writer.add_scalar(tag="train_loss", step=step, value=avg_loss_value[0])

            if step % 10 == 0:
                print(
                    "\tTrain pass {}, Step {}, Cost {}, Miou {}".format(
                        pass_id, step, avg_loss_value[0], miou_value[0]
                    )
                )

            if step % cfg.TRAIN.SNAPSHOT_BATCH == 0:
                eval_miou = 0
                # TODO: 使用metric
                auc_metric = fluid.metrics.Auc("ROC")
                test_losses = []
                test_mious = []
                for test_data in test_reader():
                    preds, test_loss, test_miou = exe_test.run(
                        test_program,
                        feed=test_feeder.feed(test_data),
                        fetch_list=[prediction, avg_loss, miou],
                    )
                    # print(test_data)
                    # auc_metric.update(preds=preds,labels=)
                    test_losses.append(test_loss[0])
                    test_mious.append(test_miou[0])

                eval_miou = np.average(np.array(test_mious))
                writer.add_scalar(
                    tag="test_loss", step=step, value=np.average(np.array(test_losses)),
                )
                print(
                    "Test loss: {} ,miou: {}".format(np.average(np.array(test_losses)), eval_miou)
                )
                print("目前最高的测试MIOU是： ", best_miou)

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("Got NaN loss, training failed.")

            if step % cfg.TRAIN.SNAPSHOT_BATCH == 0 and eval_miou > best_miou:
                best_miou = eval_miou
                print("Saving params of step: %d" % step)
                fluid.io.save_inference_model(
                    cfg.TRAIN.INF_MODEL_PATH,
                    feeded_var_names=["volume"],
                    target_vars=[prediction],
                    executor=exe,
                    main_program=train_program,
                )
                # fluid.io.save_persistables(exe, cfg.TRAIN.CKPT_MODEL, train_program)


if __name__ == "__main__":
    args = parse_args()
    main()
