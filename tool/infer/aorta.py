import argparse
import os
import sys
import math
from multiprocessing import Pool
import time
import random

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, default="./train-diameter")
args = parser.parse_args()


def score(data, split=0.2):
    if len(data) < 10:
        return -1, -1, -1, -1, -1

    mins = []
    # print("\n\n\n\n\n", data)
    for idx in range(len(data)):
        if len(data[idx]) == 1:
            continue
        mins.append(np.min(data[idx][1:]))
    # print(mins)
    abdomin = np.max(mins[: int(len(mins) * split)])
    chest = np.max(mins[int(len(mins) * split) :])
    if abdomin > 30:
        abdomin += 20
    res = max(abdomin, chest)
    if 40 < res < 50:
        cat1 = 1
    elif res > 50:
        cat1 = 2
    else:
        cat1 = 0
    if res > 39.5:
        cat2 = 1
    else:
        cat2 = 0
    if res > 50:
        cat3 = 1
    else:
        cat3 = 0
    # print(abdomin, chest, cat1, cat2, cat3)
    return abdomin, chest, cat1, cat2, cat3


if __name__ == "__main__":
    names = os.listdir(args.in_dir)
    for name in names:
        path = os.path.join(args.in_dir, name)
        with open(path, "r") as f:
            data = f.readlines()
            data = [d.split(",") for d in data[1:]]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    try:
                        data[i][j] = float(data[i][j])
                    except:
                        del data[i][j]
            print(name.split("_")[0], end="\t")
            for d in score(data):
                print(d, end="\t")
            print()
