# 在验证集上对模型的多种指标进行评估
import os
from medpy import metric
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import argparse

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
    headers = []
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
    ]
    for m in cfg.EVAL.METRICS:
        if m in metrics:
            headers.append(m)

    if not os.path.exists(os.path.dirname(cfg.EVAL.PATH.RESULT)):
        os.makedirs(os.path.dirname(cfg.EVAL.PATH.RESULT))

    preds = util.listdir(cfg.EVAL.PATH.SEG)
    labels = util.listdir(cfg.EVAL.PATH.GT)

    # print(preds, labels)
    res = []
    # BUG: pbar一直另起一行
    pbar = tqdm(total=len(preds), desc="评测进度")
    for ind in range(len(preds)):
        pbar.update()
        pbar.set_postfix(file=preds[ind])
        predf = nib.load(os.path.join(cfg.EVAL.PATH.SEG, preds[ind]))
        # TODO: 改回labels
        labf = nib.load(os.path.join(cfg.EVAL.PATH.GT, preds[ind]))

        pred = predf.get_fdata()
        lab = labf.get_fdata()
        assert pred.shape == lab.shape, "分割结果和GT大小不同： {}，{}".format(preds[ind], labels[ind])

        temp = []

        if "FP" in cfg.EVAL.METRICS:
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
            rec = metric.binary.recall(pred, lab)
            temp.append(rec)

        if "Specificity" in cfg.EVAL.METRICS:
            spec = metric.binary.specificity(pred, lab)
            temp.append(spec)

        if "Accuracy" in cfg.EVAL.METRICS:
            tp = metric.binary.true_positive_rate(pred, lab)
            tn = metric.binary.true_negative_rate(pred, lab)
            acc = tp + tn
            temp.append(acc)

        if "Kappa" in cfg.EVAL.METRICS:
            pass

        if "Dice" in cfg.EVAL.METRICS:
            dice = metric.dc(pred, lab)
            temp.append(dice)

        if "IOU" in cfg.EVAL.METRICS:
            iou = metric.binary.jc(pred, lab)
            temp.append(iou)

        res.append(temp)

    f = open(cfg.EVAL.PATH.RESULT + "-" + str(datetime.now()) + ".csv", "w")
    print("文件", end=",", file=f)
    for ind, header in enumerate(headers):
        print(header, end="," if ind != len(headers) - 1 else "\n", file=f)

    for ind in range(len(preds)):
        print(preds[ind], end=",", file=f)
        for ind, x in enumerate(res[ind]):
            print(x, end="," if ind != len(headers) - 1 else "\n", file=f)
        print("\n", file=f)

    f.close()


# TODO: 绘制箱须图
# https://matplotlib.org/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py

if __name__ == "__main__":
    parse_args()
    main()
