# 用2d平面内的数据计算管径
import argparse
import os
import sys
import math
from multiprocessing import Pool
import time
import random

from tqdm import tqdm
import numpy as np
import cv2
import nibabel as nib
import scipy.ndimage
from skimage import filters
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


parser = argparse.ArgumentParser()
parser.add_argument("--seg_dir", type=str, default="/home/lin/Desktop/git/med/med_lib/nii_inf/seg")
parser.add_argument("--dia_dir", type=str, default="./diameter")
parser.add_argument("--filter_arch", type=bool, default=True)
args = parser.parse_args()


class Polygon:
    def __init__(self, points, height, pixdim, shape=[512, 512]):
        self.idx = 0
        self.shape = shape
        self.points = points
        random.shuffle(self.points)
        self.center = [0, 0]
        self.height = height
        self.diameters = []
        self.pixdim = pixdim
        # TODO: 研究用霍夫圆算圆心
        # TODO: 解决圆心不在圆里的情况
        # 计算边缘位置平均数做圆心
        for p in self.points:
            self.center[0] += p[0]
            self.center[1] += p[1]
        self.center[0] /= len(self.points)
        self.center[1] /= len(self.points)
        self.center[0] = int(self.center[0])
        self.center[1] = int(self.center[1])
        if not self.is_inside():
            self.points = []

    def is_inside(self):
        """去掉中点不在多边形内和太小的多边形.

        Returns
        -------
        type
            Description of returned object.

        """
        if len(self.points) < 30:
            return False
        label = np.zeros(self.shape, dtype="uint8")
        for p in self.points:
            label[p[0]][p[1]] = 1
        flood_fill(label, tuple(self.center), 1, selem=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], in_place=True)

        if label.sum() > label.size / 2:
            return False
        return True

    def ang_sort(self):
        pass

    def plot(self):
        img = np.zeros([512, 512])
        for ind in range(len(self.points)):
            img[self.points[ind][0]][self.points[ind][1]] = 1
        img[self.center[0]][self.center[1]] = 1
        plt.imshow(img)
        plt.show()

    def cal_diameters(self, ang_range=[0, np.pi], split=10):
        """用类似二分的方法，平行线夹计算管径.

        y - y0 + d = k ( x - x0 )：d是这根线在y轴上移动的距离
        取x=0(a), x=1(b) a，b两点，通过判断a，b，points上的各个点p是不是都向同一个方向转，判断直线是不是已经移出了多边形

        Parameters
        ----------
        ang_range : list
            直线和x轴角度的范围.
        split : int
            在这个范围内，平均测量多少个方向.
        pixdim : float
            片子的pixdim，从像素换算到实际的mm.

        Returns
        -------
        list
            ang_range 角度范围内，split 等分个方向上，平行线夹的管径是多少mm.

        """
        if len(self.points) == 0:
            self.diameters = 0
            print("[Error] Polygon at height {} contains no point".format(self.height))
            return

        def is_right(a, b, c):
            a = np.array((b[0] - a[0], b[1] - a[1]))
            b = np.array((c[0] - b[0], c[1] - b[1]))
            res = np.cross(a, b)
            return res >= 0

        # print(self.height)
        self.diameters = []
        center = self.center
        for alpha in np.arange(ang_range[0], ang_range[1], (ang_range[1] - ang_range[0]) / split):
            if alpha == np.pi / 2:
                continue
            k = math.tan(alpha)

            def binary_search(step):
                d = 0
                prev_out = False
                while True:
                    # print(d)
                    ya = center[1] - d + k * (0 - center[0])  # (0,ya)
                    yb = center[1] - d + k * (1 - center[0])  # (1,yb)
                    dir = is_right((0, ya), (1, yb), self.points[0])
                    same_dir = True
                    for p in self.points:
                        if dir != is_right((0, ya), (1, yb), p):
                            same_dir = False
                            break
                    # print(same_dir)
                    if same_dir:
                        if not prev_out:
                            step /= 3
                        d -= step
                        prev_out = True
                    else:
                        d += step
                        prev_out = False
                    if abs(step) < 0.1:
                        break
                return d

            self.diameters.append((binary_search(40) - binary_search(-40)) * np.abs(np.cos(alpha)) * self.pixdim)
        # print(self.diameters)
        return self.idx, self.height, self.diameters


def dist(a, b):
    h = a.height - b.height
    ca = a.center
    cb = b.center
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2 + h ** 2) ** 0.5


# print(dist(Polygon([[0, 0]], 0, [0, 0]), Polygon([[0, 0]], 1, [1, 2])))


def blood_sort(polygons):
    """按照血流方向反向对血管中的圆进行排序.

    Parameters
    ----------
    polygons : list
        一个血管所有片曾polygon的list.

    Returns
    -------
    list
        按照血流反向排序的polygon list.

    """
    if len(polygons) == 0:
        print("Error, got None points")
        return

    # 1. 计算最低的片层，认为是降主动脉最下面一片，排序的开始
    ordered = []
    min_height = polygons[0].height
    min_ind = 0
    for idx, p in enumerate(polygons):
        if p.height < min_height:
            min_height = p
            min_ind = idx
    # 2. 一直找无序的list中和当前最后一个距离最近的作为下一个
    ordered.append(polygons[min_ind])
    del polygons[min_ind]

    while len(polygons) != 0:
        min_dist = dist(ordered[-1], polygons[0])
        min_ind = 0
        for idx, p in enumerate(polygons):
            d = dist(ordered[-1], p)
            if d < min_dist:
                min_dist = d
                min_ind = idx
        ordered.append(polygons[min_ind])
        del polygons[min_ind]
    for idx, p in enumerate(ordered):
        p.idx = idx
    return ordered


def cal(polygon):
    return polygon.cal_diameters()


def cal_diameter(seg_path, filter_arch, dia_dir):
    """计算seg_path这个nii分割文件的所有管径，返回.

    过程：
    1.  按照血流反向，获取所有圆
    1.1 按照高度分片层，层内找连通块，可能一块可能两块，计算层中心
    1.2 从最下面的中心开始，找最近的没入序列的中心，对中心按照血流方向反向排序

    2. 用平行线夹计算血管管径
    2.1

    Parameters
    ----------
    seg_path : str
        这个人的分割文件路径.

    Returns
    -------
    type
        Description of returned object.

    """
    print(seg_path)
    start = int(time.time())
    segf = nib.load(seg_path)
    seg_data = segf.get_fdata()
    pixdim = segf.header["pixdim"][1]
    seg_data[seg_data > 0.9] = 1
    seg_data = seg_data.astype("uint8")
    print(seg_data.shape)
    polygons = []
    for height in range(seg_data.shape[2]):
        label = seg_data[:, :, height]
        label = filters.roberts(label)
        vol, num = scipy.ndimage.label(label, np.ones([3, 3]))
        for label_idx in range(1, num + 1):
            xs, ys = np.where(vol == label_idx)
            points = []
            for x, y in zip(xs, ys):
                points.append([int(x), int(y)])
            polygons.append(Polygon(points, height, pixdim, label.shape))
    polygons = [p for p in polygons if len(p.points) != 0]
    polygons = blood_sort(polygons)

    pool = Pool(8)
    diameters = []
    for res in tqdm(pool.imap_unordered(cal, polygons), total=len(polygons)):
        diameters.append(res)
    pool.close()
    pool.join()

    diameters = sorted(diameters, key=lambda x: x[0])
    for d, p in zip(diameters, polygons):
        p.diameters = d[2]

    # # 顺序进行
    # for p in tqdm(polygons):
    #     p.cal_diameters()

    print(os.path.join(dia_dir, seg_path.split("/")[-1]))
    f = open(os.path.join(dia_dir, seg_path.split("/")[-1].rstrip(".gz").rstrip(".nii")) + ".csv", "w")
    print((int(time.time()) - start) / 60)
    print((int(time.time()) - start) / 60, end="\n", file=f)
    # for d in diameters:
    #     print(d[1], end=",", file=f)
    #     for data in d[2]:
    #         print(data, end=",", file=f)
    #     print(file=f)
    #
    for p in polygons:
        print(p.height, end=",", file=f)
        # print(p.diameters)
        if filter_arch:
            if np.max(p.diameters) > np.min(p.diameters) * 2:
                continue
        for d in p.diameters:
            print(d, end=",", file=f)
        print(end="\n", file=f)
    f.close()


if __name__ == "__main__":
    names = os.listdir(args.seg_dir)
    for name in names:
        if os.path.exists(os.path.join(args.dia_dir, name.replace(".nii.gz", ".csv"))):
            names.remove(name)
    print(names)
    print("{} patients to measure in total.".format(len(names)))

    start = int(time.time())
    tot = len(names)
    count = 0

    for name in names:
        cal_diameter(os.path.join(args.seg_dir, name), args.filter_arch, args.dia_dir)
        count += 1
        print(
            "\t\tFinished {}/{}, expected to finish in {} minutes".format(
                count, tot, int(time.time() - start) / count * (tot - count) / 60
            )
        )
