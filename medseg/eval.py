# 在验证集上对模型的多种指标进行评估
import os
from multiprocessing import Pool, cpu_count
from datetime import datetime
import argparse

from medpy import metric
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage

import utils.util as util
from utils.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="预测")
    parser.add_argument("-c", "--cfg_file", type=str, help="配置文件路径")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)

    cfg.set_immutable(True)


def main():
    headers = []
    # TODO: 研究一个更简洁的保持第一行和下面的数顺序一致的方法
    metrics = [
        "FP",
        "FN",
        "TP",
        "TN",
        "Precision",
        "Recall",
        "Sensitivity",
        "Specificity",
        "Accuracy",
        "Kappa",
        "Dice",
        "IOU",
        # 3D
        "Assd",
        "Ravd",
    ]
    for m in metrics:
        if m in cfg.EVAL.METRICS:
            headers.append(m)

    preds = util.listdir(cfg.EVAL.PATH.SEG)
    labels = util.listdir(cfg.EVAL.PATH.GT)

    f = open(cfg.EVAL.PATH.NAME + "-" + str(datetime.now()) + ".csv", "w")
    print("文件", end=",", file=f)
    for ind, header in enumerate(headers):
        print(header, end="," if ind != len(headers) - 1 else "\n", file=f)
    with Pool(cpu_count()) as p:
        res = p.map(calculate, [(preds[idx], labels[idx]) for idx in range(len(preds))])
    for pred, r in res:
        print(pred, end=",", file=f)
        for ind, x in enumerate(r):
            print(x, end="," if ind != len(headers) - 1 else "\n", file=f)
    f.close()


# def write_res(pred, res):
#     f = open(cfg.EVAL.PATH.RESULT + "-" + str(datetime.now()) + ".csv", "w+")
#     print(pred, end=",", file=f)
#     for ind, x in enumerate(res[ind]):
#         print(x, end="," if ieval.csv-2020-12-16 18:59:40.731539.csvnd != len(headers) - 1 else "\n", file=f)
#     print("\n", file=f)
#     f.close()


def calculate(input):
    pred_name = input[0]
    lab_name = input[1]

    predf = nib.load(os.path.join(cfg.EVAL.PATH.SEG, pred_name))
    labf = nib.load(os.path.join(cfg.EVAL.PATH.GT, lab_name))

    pred = predf.get_fdata()
    lab = labf.get_fdata()
    print(pred_name, lab_name, pred.shape, lab.shape)
    if lab.shape[0] != pred.shape[0]:
        ratio = [a / b for a, b in zip(pred.shape, lab.shape)]
        lab = scipy.ndimage.interpolation.zoom(lab, ratio, order=1)
        print("插值后大小：　", pred.shape, lab.shape)
    assert pred.shape == lab.shape, "分割结果和GT大小不同： {}，{}, {}".format(
        pred.shape, lab.shape, preds[ind]
    )

    temp = []
    if "FP" in cfg.EVAL.METRICS:
        # fp = metric.binary.obj_fpr(pred, lab)
        # temp.append(fp)
        pass

    if "FN" in cfg.EVAL.METRICS:
        pass

    if "TP" in cfg.EVAL.METRICS:
        tp = metric.binary.true_positive_rate(pred, lab)
        temp.append(tp)

    if "TN" in cfg.EVAL.METRICS:
        tn = metric.binary.true_negative_rate(pred, lab)
        temp.append(tn)

    if "Precision" in cfg.EVAL.METRICS:
        prec = metric.binary.precision(pred, lab)
        temp.append(prec)

    if "Recall" in cfg.EVAL.METRICS:
        rec = metric.binary.recall(pred, lab)
        temp.append(rec)

    if "Sensitivity" in cfg.EVAL.METRICS:
        rec = metric.binary.sensitivity(pred, lab)  # same as recall
        temp.append(rec)

    if "Specificity" in cfg.EVAL.METRICS:
        spec = metric.binary.specificity(pred, lab)
        temp.append(spec)

    if "Accuracy" in cfg.EVAL.METRICS:
        tp = metric.binary.true_positive_rate(pred, lab)
        tn = metric.binary.true_negative_rate(pred, lab)
        acc = (tp + tn) / 2
        temp.append(acc)

    if "Kappa" in cfg.EVAL.METRICS:
        pass

    if "Dice" in cfg.EVAL.METRICS:
        dice = metric.dc(pred, lab)
        temp.append(dice)

    if "IOU" in cfg.EVAL.METRICS:
        iou = metric.binary.jc(pred, lab)
        temp.append(iou)

    if "Assd" in cfg.EVAL.METRICS:
        assd = metric.binary.assd(pred, lab)
        temp.append(assd)

    if "Ravd" in cfg.EVAL.METRICS:
        ravd = metric.binary.ravd(pred, lab)
        temp.append(ravd)

    return pred_name, temp


# TODO: 绘制箱须图
# https://matplotlib.org/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py

if __name__ == "__main__":
    parse_args()
    main()
