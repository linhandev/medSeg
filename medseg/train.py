from __future__ import print_function

import sys
import os
import argparse

import math
import numpy as np

from datetime import datetime

import paddle
import paddle.fluid as fluid
from paddle.utils.plot import Ploter
from paddle.fluid.layers import log

from visualdl import LogWriter

from models.unet_simple import unet_simple
from models.deeplabv3p import deeplabv3p
from models.unet_base import unet_base
from util import *
import random

from config import *

from loss import *
import aug
from tqdm import tqdm
from lib.threshold_function_module import windowlize_image


def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="是否用GPU")
    parser.add_argument("--num_epochs", type=int, default=20, help="多少个epoch")
    parser.add_argument("--net", type=str, default="unet_base", help="选择使用网络类型")
    parser.add_argument("--do_aug", action="store_true", default=False, help="是否进行数据增强")
    parser.add_argument("--windowlize", action="store_true", default=False, help="是否进行窗口化")
    args = parser.parse_args()
    return args


def data_reader(part_start=0, part_end=8, is_test=False):
    data_names = listdir(preprocess_path)  # listdir是会对数据排序的，多次调用顺序相同，训练集和测试集划分完才shuffle，能保证训练集和测试集没有重合
    data_part = data_names[len(data_names) * part_start // 10 : len(data_names) * part_end // 10]
    random.shuffle(data_part)

    def reader():
        for data_name in tqdm(data_part):
            data = np.load(os.path.join(preprocess_path, data_name))
            vol = data[0:3, :, :].reshape(3, 512, 512).astype("float32")
            lab = data[3, :, :].reshape(1, 512, 512).astype("int32")
            if args.windowlize:
                vol = windowlize_image(vol, 200, 70)  # 肝脏常用
            yield vol, lab

    return reader


def aug_mapper(data):
    vol = data[0]
    lab = data[1]
    if args.do_aug:  # 注意不要增强第0维,那是厚度的方向
        vol, lab = aug.flip(vol, lab, [0.5, 0.5, 0])
        vol, lab = aug.rotate(vol, lab, [0, [-15, 15], 0], [0, 0.6, 0])
        vol, lab = aug.zoom(vol, lab, [[1, 1], [0.8, 1], [0.8, 1]], [0, 0.3, 0.3])
        vol, lab = aug.crop(vol, lab, [3, 512, 512])

        # print("crop", vol.shape, lab.shape)
    assert vol.shape == (3, 512, 512) and lab.shape == (1, 512, 512), "数据维度错误,volume {}, label {}".format(vol.shape, lab.shape)
    assert len(np.where((lab != 0) & (lab != 1))[0]) == 0, "数据中有0,1之外的数{}".format(np.where((lab != 0) & (lab != 1)))
    return vol, lab


def main():
    batch_size = 16

    # 训练ckpt和inf模型路径
    param_base_dir = os.path.join(code_base_dir, "params")
    param_base_dir = os.path.join(param_base_dir, args.net)
    infer_param_path = os.path.join(param_base_dir, "inf")
    ckpt_param_path = os.path.join(param_base_dir, "ckpt")
    print(infer_param_path)
    print(ckpt_param_path)

    test_reader = paddle.batch(data_reader(8, 10, True), batch_size)

    train_program = fluid.Program()
    train_init = fluid.Program()

    with fluid.program_guard(train_program, train_init):
        image = fluid.layers.data(name="image", shape=[3, 512, 512], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1, 512, 512], dtype="int32")

        if args.net == "unet_simple":
            prediction = unet_simple(image, 2, [512, 512])
        elif args.net == "unet_base":
            prediction = unet_base(image, 2, [512, 512])
        elif args.net == "deeplabv3":
            prediction = deeplabv3p(image, 2)
        else:
            print("错误的网络类型")
            sys.exit(0)

        avg_loss = create_loss(prediction, label, 2)

        miou = mean_iou(prediction, label, 2)

        decay = paddle.fluid.regularizer.L2Decay(0.0001)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.003, regularization=decay)

        # optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.006, momentum=0.8,regularization=decay)

        optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_init)

    # fluid.io.load_persistables(exe, ckpt_param_path, train_init)

    exe_test = fluid.Executor(place)

    test_program = train_program.clone(for_test=True)

    train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)
    # test_program = fluid.CompiledProgram(test_program).with_data_parallel(loss_name=avg_loss.name)

    train_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    test_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    step = 1
    best_miou = 0

    train_prompt = "Train_miou"
    test_prompt = "Test_miou"

    # plot_prompt = Ploter(train_prompt, test_prompt)
    writer = LogWriter(logdir="../log/{}".format(datetime.now()))


    train_reader = fluid.io.xmap_readers(aug_mapper, data_reader(0, 8), 8, 32, False)

    step = 0
    eval_step = 1000
    for pass_id in range(args.num_epochs):
        batch = []
        for record in train_reader():
            batch.append(record)
            if len(batch) == batch_size:
                step += 1
                avg_loss_value, miou_value = exe.run(train_program, feed=train_feeder.feed(batch), fetch_list=[avg_loss, miou])
                batch = []
                writer.add_scalar(tag="train_loss", step=step, value=avg_loss_value[0])

                if step % 10 == 0:
                    print("\t\tTrain pass {}, Step {}, Cost {}, Miou {}".format(pass_id, step, avg_loss_value[0], miou_value[0]))

                eval_miou = 0
                if step % eval_step == 0:
                    auc_metric = fluid.metrics.Auc("AUC")
                    test_losses = []
                    test_mious = []
                    for test_data in test_reader():
                        _, test_loss, test_miou = exe_test.run(
                            test_program, feed=test_feeder.feed(test_data), fetch_list=[prediction, avg_loss, miou]
                        )
                        test_losses.append(test_loss[0])
                        test_mious.append(test_miou[0])

                    eval_miou = np.average(np.array(test_mious))
                    writer.add_scalar(tag="test_loss", step=step/eval_step, value=np.average(np.array(test_losses)))
                    print("Test loss: {} ,miou: {}".format(np.average(np.array(test_losses)), eval_miou))

                if math.isnan(float(avg_loss_value[0])):
                    sys.exit("got NaN loss, training failed.")

                if step % eval_step == 0 and eval_miou > best_miou:
                    best_miou = eval_miou
                    print("Saving params of step: %d" % step)
                    fluid.io.save_inference_model(
                        infer_param_path,
                        feeded_var_names=["image"],
                        target_vars=[prediction],
                        executor=exe,
                        main_program=train_program,
                    )
                    # fluid.io.save_persistables(exe, ckpt_param_path, train_program)
        print(best_miou)


if __name__ == "__main__":
    args = parse_args()
    main()
