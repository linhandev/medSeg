# coding=utf-8
import sys
import os
import argparse
import random
import math
from datetime import datetime
import multiprocessing

import numpy as np
from tqdm.auto import tqdm
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import log
from visualdl import LogWriter


import utils.util as util
from utils.config import cfg
import loss
import aug
from models.model import create_model


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
    # TODO: 打印cfg配置


npz_names = util.listdir(cfg.TRAIN.DATA_PATH)
random.shuffle(npz_names)


def data_reader(part_start=0, part_end=8):
    # NOTE: 这种分法效率高好写，但是npz很少的时候分得不准。npz文件至少分10个
    npz_part = npz_names[int(len(npz_names) * part_start / 10) : int(len(npz_names) * part_end / 10)]

    def reader():
        # BUG: tqdm每次更新都另起一行，此外要单独测试windows上好不好使
        if cfg.TRAIN.DATA_COUNT != -1:
            pbar = tqdm(total=cfg.TRAIN.DATA_COUNT, desc="训练进度")
        for npz_name in npz_part:
            data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz_name))
            imgs = data["imgs"]
            labs = data["labs"]
            assert len(np.where(labs == 1)[0]) + len(np.where(labs == 0)[0]) == labs.size, "非法的label数值"
            if cfg.AUG.WINDOWLIZE:
                imgs = util.windowlize_image(imgs, cfg.AUG.WWWC)
            else:
                imgs = util.windowlize_image(imgs, (4096, 0))
            inds = [x for x in range(imgs.shape[0])]
            random.shuffle(inds)
            for ind in inds:
                if cfg.TRAIN.DATA_COUNT != -1:
                    pbar.update()
                vol = imgs[ind].reshape(cfg.TRAIN.THICKNESS, 512, 512).astype("float32")
                lab = labs[ind].reshape(1, 512, 512).astype("int32")
                yield vol, lab
        # TODO: 标签平滑
        # https://medium.com/@lessw/label-smoothing-deep-learning-google-brain-explains-why-it-works-and-when-to-use-sota-tips-977733ef020

    return reader


def aug_mapper(data):
    vol = data[0]
    lab = data[1]
    ww, wc = cfg.AUG.WWWC
    # NOTE: 注意不要增强第0维,那是厚度的方向
    vol, lab = aug.flip(vol, lab, cfg.AUG.FLIP.RATIO)
    vol, lab = aug.rotate(vol, lab, cfg.AUG.ROTATE.RANGE, cfg.AUG.ROTATE.RATIO, wc - ww / 2)
    vol, lab = aug.zoom(vol, lab, cfg.AUG.ZOOM.RANGE, cfg.AUG.ZOOM.RATIO)
    vol, lab = aug.crop(vol, lab, cfg.AUG.CROP.SIZE, wc - ww / 2)
    return vol, lab


def main():
    train_program = fluid.Program()
    train_init = fluid.Program()

    with fluid.program_guard(train_program, train_init):
        image = fluid.layers.data(name="image", shape=[cfg.TRAIN.THICKNESS, 512, 512], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1, 512, 512], dtype="int32")
        train_loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=cfg.TRAIN.BATCH_SIZE * 2,
            iterable=True,
            use_double_buffer=True,
        )
        prediction = create_model(image, 2)
        avg_loss = loss.create_loss(prediction, label, 2)
        miou = loss.mean_iou(prediction, label, 2)

        # 进行正则化
        if cfg.TRAIN.REG_TYPE == "L1":
            decay = paddle.fluid.regularizer.L1Decay(cfg.TRAIN.REG_COEFF)
        elif cfg.TRAIN.REG_TYPE == "L2":
            decay = paddle.fluid.regularizer.L2Decay(cfg.TRAIN.REG_COEFF)
        else:
            decay = None

        # 选择优化器
        lr = fluid.layers.piecewise_decay(boundaries=cfg.TRAIN.BOUNDARIES, values=cfg.TRAIN.LR)
        if cfg.TRAIN.OPTIMIZER == "adam":
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, regularization=decay,)
        elif cfg.TRAIN.OPTIMIZER == "sgd":
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=lr, regularization=decay)
        elif cfg.TRAIN.OPTIMIZE == "momentum":
            optimizer = fluid.optimizer.Momentum(momentum=0.9, learning_rate=lr, regularization=decay,)
        else:
            raise Exception("错误的优化器类型: {}".format(cfg.TRAIN.OPTIMIZER))
        optimizer.minimize(avg_loss)

    places = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
    exe = fluid.Executor(places)
    exe.run(train_init)
    exe_test = fluid.Executor(places)

    test_program = train_program.clone(for_test=True)
    compiled_train_program = fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)

    if cfg.TRAIN.PRETRAINED_WEIGHT != "":
        print("Loading paramaters")
        fluid.io.load_persistables(exe, cfg.TRAIN.PRETRAINED_WEIGHT, train_program)

    # train_reader = fluid.io.xmap_readers(
    #     aug_mapper, data_reader(0, 8), multiprocessing.cpu_count()/2, 16
    # )
    train_reader = data_reader(0, 8)
    train_loader.set_sample_generator(train_reader, batch_size=cfg.TRAIN.BATCH_SIZE, places=places)
    test_reader = paddle.batch(data_reader(8, 10), cfg.INFER.BATCH_SIZE)
    test_feeder = fluid.DataFeeder(place=places, feed_list=[image, label])

    writer = LogWriter(logdir="/home/aistudio/log/{}".format(datetime.now()))

    step = 0
    best_miou = 0

    for pass_id in range(cfg.TRAIN.EPOCHS):
        for train_data in train_loader():
            step += 1
            avg_loss_value, miou_value = exe.run(
                compiled_train_program, feed=train_data, fetch_list=[avg_loss, miou]
            )
            writer.add_scalar(tag="train_loss", step=step, value=avg_loss_value[0])
            writer.add_scalar(tag="train_miou", step=step, value=miou_value[0])
            if step % cfg.TRAIN.DISP_BATCH == 0:
                print(
                    "\tTrain pass {}, Step {}, Cost {}, Miou {}".format(
                        pass_id, step, avg_loss_value[0], miou_value[0]
                    )
                )

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("Got NaN loss, training failed.")

            if step % cfg.TRAIN.SNAPSHOT_BATCH == 0 and cfg.TRAIN.DO_EVAL:
                test_step = 0
                eval_miou = 0
                test_losses = []
                test_mious = []
                for test_data in test_reader():
                    test_step += 1
                    preds, test_loss, test_miou = exe_test.run(
                        test_program,
                        feed=test_feeder.feed(test_data),
                        fetch_list=[prediction, avg_loss, miou],
                    )
                    test_losses.append(test_loss[0])
                    test_mious.append(test_miou[0])
                    if test_step % cfg.TRAIN.DISP_BATCH == 0:
                        print("\t\tTest Loss: {} , Miou: {}".format(test_loss[0], test_miou[0]))

                eval_miou = np.average(np.array(test_mious))
                writer.add_scalar(
                    tag="test_miou", step=step, value=eval_miou,
                )
                print("Test loss: {} ,miou: {}".format(np.average(np.array(test_losses)), eval_miou))
                ckpt_dir = os.path.join(cfg.TRAIN.CKPT_MODEL_PATH, str(step) + "_" + str(eval_miou))
                fluid.io.save_persistables(exe, ckpt_dir, train_program)

                print("此前最高的测试MIOU是： ", best_miou)

            if step % cfg.TRAIN.SNAPSHOT_BATCH == 0 and eval_miou > best_miou:
                best_miou = eval_miou
                print("正在保存第 {} step的权重".format(step))
                fluid.io.save_inference_model(
                    cfg.TRAIN.INF_MODEL_PATH,
                    feeded_var_names=["image"],
                    target_vars=[prediction],
                    executor=exe,
                    main_program=train_program,
                )


if __name__ == "__main__":
    args = parse_args()
    main()
