import os
import logging

import nibabel as nib
import numpy as np
import cv2
import scipy.ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh
from pypinyin import pinyin, Style


logging.basicConfig(level=logging.NOTSET)

def listdir(path):
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")
    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs

def nii2png_single(nii_path, png_folder, rot=1, wwwl=(256, 0), islabel=False):
    """将一个nii扫描转换成一系列图片，并进行简单的检查.
    # TODO: 检查是否只有一个连通块
    # TODO: 检查是否只有一种前景
    Parameters
    ----------
    nii_path : str
        nii扫描文件的路径.
    png_path : type
        图片存在哪个文件夹.
    rot : int
        扫描进行几次旋转，摆正体位.
    wwwl : (int, int)
        窗宽窗位.
    """
    volf = nib.load(nii_path)
    nii_name = os.path.basename(nii_path)
    vol = volf.get_fdata()
    if vol.shape[0] == 1024:
        vol = scipy.ndimage.interpolation.zoom(vol, [0.5, 0.5, 1], order=1 if islabel else 3)
    for _ in range(rot):
        vol = np.rot90(vol)
    if not islabel:
        # vol = vol + np.random.randn() * 50 - 10
        wl, wh = (wwwl[1] - wwwl[0] / 2, wwwl[1] + wwwl[0] / 2)
        vol = vol.astype("float32").clip(wl, wh)
        vol = (vol - wl) / (wh - wl) * 256
    vol = vol.astype("uint8")
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    print(nii_name)
    print(nii_name[:-7])

    for ind in range(1, vol.shape[2] - 1):
        file_path = os.path.join(
            png_folder,
            "{}-{}.png".format(nii_name[:-7], ind),
        )
        if islabel:
            slice = vol[:, :, ind]
        else:
            slice = vol[:, :, ind - 1 : ind + 2]

        cv2.imwrite(file_path, slice)


def nii2png_folder(nii_folder, png_folder, rot=1, wwwl=(400, 0), subfolder=False, islabel=False):
    """将一个文件夹里所有的nii转换成png.
    Parameters
    ----------
    nii_folder : str
        放所有nii的文件夹.
    png_folder : str
        放所有png的文件夹.
    rot : int
        旋转几次.
    wwwl : (int, int)
        窗宽窗位.
    subfolder : bool
        是否给每一个nii创建单独的文件夹.
    """
    nii_names = os.listdir(nii_folder)
    for nii_name in tqdm(nii_names):
        if subfolder:
            png_folder = os.path.join(png_folder, nii_name)
        if len(nii_name) > 12:
            nii2png_single(os.path.join(nii_folder, nii_name), png_folder, 1, wwwl, islabel)
        else:
            nii2png_single(os.path.join(nii_folder, nii_name), png_folder, 3, wwwl, islabel)
        # print(len(nii_name))
        # print(nii_name)
        # input("here")
        # os.system("rm /home/lin/Desktop/data/aorta/dataset/scan/*")


def check_nii_match(scan_dir, label_dir):
    """检查两个目录下的扫描和标签是不是对的上.
    Parameters
    ----------
    scan_dir : str
        扫描所在路径.
    label_dir : str
        标签所在路径.
    Returns
    -------
    bool
        比较的结果，对上了返回True，否则False,具体的细节直接打到stdio.
    """
    pass_check = True
    scans = os.listdir(scan_dir)
    labels = os.listdir(label_dir)
    scans = [n for n in scans if n.endswith("nii") or n.endswith("gz")]
    labels = [n for n in labels if n.endswith("nii") or n.endswith("gz")]

    scan_names = [n.rstrip(".gz").rstrip(".nii") for n in scans]
    label_names = [n.rstrip(".gz").rstrip(".nii") for n in labels]

    if len(scans) != len(labels):
        logging.error("Number of scnas({}) and labels ({}) don't match".format(len(scans), len(labels)))
        pass_check = False
    else:
        logging.info("Pass file number check")

    names_match = True
    for ind, s in enumerate(scan_names):
        if s not in label_names:
            logging.error("Scan {} dont have corresponding label".format(s))
            names_match = False
            print("removing {}".format(s))
            os.remove(os.path.join(scan_dir, scans[ind]))

    for l in label_names:
        if l not in scan_names:
            logging.error("Label {} dont have corresponding scan".format(l))
            names_match = False

    if names_match:
        logging.info("Pass file names check")
    else:
        pass_check = False

    scans = os.listdir(scan_dir)
    labels = os.listdir(label_dir)
    scans = [n for n in scans if n.endswith("nii") or n.endswith("gz")]
    labels = [n for n in labels if n.endswith("nii") or n.endswith("gz")]

    scan_names = [n.rstrip(".gz").rstrip(".nii") for n in scans]
    label_names = [n.rstrip(".gz").rstrip(".nii") for n in labels]

    for scan_name in scans:
        scanf = nib.load(os.path.join(scan_dir, scan_name))
        labelf = nib.load(os.path.join(label_dir, scan_name))
        if (scanf.affine == np.eye(4)).all():
            logging.warn("Scan {} have np.eye(4) affine, check the header".format(scan_name))
        if (labelf.affine == np.eye(4)).all():
            logging.warn("Label {} have np.eye(4) affine, check the header".format(scan_name))
        if not (labelf.header["dim"] == scanf.header["dim"]).all():
            logging.error(
                "Label and scan dimension mismatch for {}, scan is {}, label is {}".format(
                    scan_name, scanf.header["dim"][1:4], labelf.header["dim"][1:4]
                )
            )
            pass_check = False
    return pass_check


def inspect_pair(scan_path, label_path):

    # 如果是nii格式
    # TODO: 完善
    if scan_path.endswith("nii") or scan_path.endswith("gz"):
        pass
    # TODO: 一对图片放到一个frame
    if scan_path.endswith("png"):
        scan = cv2.imread(scan_path)
        label = cv2.imread(label_path)
        plt.imshow(scan)
        plt.show()
        label = label * 255
        plt.imshow(label)
        plt.show()


def is_right(a, b, c):
    a = np.array((b[0] - a[0], b[1] - a[1]))
    b = np.array((c[0] - b[0], c[1] - b[1]))
    res = np.cross(a, b)
    if res >= 0:
        return True
    return False


class Polygon:
    points = []
    center = []
    height = 0
    base = []
    epsilon = 1e-6

    def __init__(self, p):
        if len(p) == 0:
            raise RuntimeError("Nan points in polygon")
        self.points = p
        if len(self.points[0]) == 2:
            points = []
            for p in self.points:
                points.append(list(p[0]))
            unique = []
            for p in points:
                if p not in unique:
                    unique.append(p)
            self.points = unique
            self.cal_rep()
            self.ang_sort()
            return

        for ind in range(len(self.points)):
            self.points[ind] = list(self.points[ind])

        self.cal_rep()
        self.ang_sort()
        try:
            self.height = p[0][2]
        except:
            print(p)

    def cal_rep(self):
        """计算中心点和基点.
        两个操作有顺序
        Returns
        -------
        type
            Description of returned object.
        """
        # print("___", np.min(self.points, axis=0))
        # print("---", np.max(self.points, axis=0))
        self.center = list((np.min(self.points, axis=0) + np.max(self.points, axis=0)) / 2)
        # print(self.center)
        # input("here")
        self.points.sort()
        self.base = self.points[0]
        del self.points[0]

    def ang_sort(self):
        """对所有的点进行极角排序.
        Returns
        -------
        type
            Description of returned object.
        """

        def cmp(a):
            return math.atan((a[1] - self.base[1]) / (a[0] - self.base[0] + self.epsilon))

        self.points.sort(key=cmp, reverse=True)

    def cal_size(self):
        """给一个数组的点，求它构成的多边形的面积.
        注意这里没有做极角排序，点本身需要满足顺时针或者逆时针顺序
        Parameters
        ----------
        points : type
            Description of parameter `points`.
        Returns
        -------
        type
            Description of returned object.
        """
        points = self.points
        tot = 0
        p = self.base
        for ind in range(0, len(points) - 1):
            a = [t2 - t1 for t1, t2 in zip(p, points[ind])]
            b = [t2 - t1 for t1, t2 in zip(p, points[ind + 1])]
            a = np.array(a)
            b = np.array(b)
            # b = b.reshape(b.size, 1)
            # print(a, b)
            res = np.cross(a, b)
            tot += (res[0] ** 2 + res[1] ** 2 + res[2] ** 2) ** (1 / 2) / 2
        return tot

    def to_2d(self):
        def rot_to_horizontal(p):
            """将一个点旋转到水平面上.
            Parameters
            ----------
            p : type
                Description of parameter `p`.
            Returns
            -------
            type
                Description of returned object.
            """
            # TODO: 在基本和y轴平行的时候会有div0错误
            epsilon = 1e-6
            if p[0] == 0 and p[1] == 0 and p[2] == 0:
                return [0, 0]
            x = p[0]
            y = p[1]
            z = p[2]
            angle = math.atan(z / ((x ** 2 + y ** 2) ** (1 / 2) + epsilon))
            unit = (y / ((x ** 2 + y ** 2) ** (1 / 2) + epsilon), -x / ((x ** 2 + y ** 2) ** (1 / 2) + epsilon), 0)
            matrix = trimesh.transformations.rotation_matrix(-angle, unit, (0, 0, 0))
            p.append(1)
            p = np.array(p).reshape([1, 4])
            p = p.transpose()
            res = np.dot(matrix, p)
            return [float(res[0]), float(res[1])]

        self.points = [[p[0] - self.base[0], p[1] - self.base[1], p[2] - self.base[2]] for p in self.points]
        print("+_+", self.center)
        self.center = [b - a for a, b in zip(self.base, self.center)]
        self.center = rot_to_horizontal(self.center)
        print("_+_", self.center)
        self.points = [rot_to_horizontal(p) for p in self.points]
        self.base = [0, 0]

    def cal_diameter(self, ang_range=[0, np.pi], split=30, pixdim=1, step=1):
        """计算这个多边形的直径
        1. 将所有点旋转到一个平面内
        2. 将半个圆周分成split份，在center做两条线，分别往向上和线下的方向运动
        3. 这个线第一次让所有多边形端点都在线一侧停止运动
        4. 计算两根线距离，作为直径
        Parameters
        ----------
        ang_range : type
            Description of parameter `ang_range`.
        split : type
            Description of parameter `split`.
        pixdim : type
            Description of parameter `pixdim`.
        step : type
            Description of parameter `step`.
        Returns
        -------
        type
            Description of returned object.
        """
        self.to_2d()
        # self.plot_2d()
        center = self.center
        diameters = [self.height]
        for alpha in np.arange(ang_range[0], ang_range[1], (ang_range[1] - ang_range[0]) / split):
            # TODO: 如果这个线是垂直的
            if alpha == np.pi / 2:
                continue
            k = math.tan(alpha)
            d = 0
            d1 = 0
            d2 = 0
            while True:
                # print("+:", d)
                y0 = center[1] - d + k * (0 - center[0])
                y1 = center[1] - d + k * (1 - center[0])
                dir = is_right((0, y0), (1, y1), self.points[0])
                same_dir = True
                for p in self.points:
                    tmp = is_right((0, y0), (1, y1), p)
                    if tmp != dir:
                        same_dir = False
                        break
                if same_dir:
                    d1 = d
                    break
                d += step

            d = 0
            while True:
                # print("-:", d)
                y0 = center[1] - d + k * (0 - center[0])
                y1 = center[1] - d + k * (1 - center[0])
                dir = is_right((0, y0), (1, y1), self.points[0])
                same_dir = True
                for p in self.points:
                    tmp = is_right((0, y0), (1, y1), p)
                    if tmp != dir:
                        same_dir = False
                        break
                if same_dir:
                    d2 = d
                    break
                d += step
            diameters.append((d1 + d2) * np.abs(np.cos(alpha)) * pixdim)
            # print("k: ", k)

        return diameters

    def plot_2d(self):
        plt.scatter([p[0] for p in self.points], [p[1] for p in self.points])
        # plt.plot([self.center[0]], [self.center[1]])
        plt.show()


"""
y-y0=k(x-x0)+b
x=0, y=y0+k(x-x0)+b
"""
# po = Polygon([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
# print(po.cal_diameter(split=2))


def dist(a, b):
    res = 0
    for p, q in zip(a, b):
        res += (p - q) ** 2
    res = res ** (1 / 2)
    return res


# print(dist([1, 0, 1], [0, 0, 3]))


def filter_polygon(points, center, thresh=1):
    """给一堆点，不一定是一个多边形，返回中心离center比较近的那个多边形.
    Parameters
    ----------
    points : type
        Description of parameter `points`.
    Returns
    -------
    type
        Description of returned object.
    """
    # TODO: 点需要极角排序

    curr_point = points[-1]
    polygons = [[]]
    p_ind = 0
    while len(points) != 0:
        found = False
        for ind in range(len(points) - 1, -1, -1):
            if dist(curr_point, points[ind]) < thresh:
                curr_point = points[ind]
                del points[ind]
                polygons[p_ind].append(curr_point)
                found = True
                break
        if not found:
            polygons.append([])
            p_ind += 1
            curr_point = points[-1]
    if center == "all":
        return polygons
    centers = []
    for polygon in polygons:
        ps = np.array(polygon)
        centers.append(np.mean(ps, axis=0))
    dists = [dist(p, center) for p in centers]
    min_ind = np.argmin(dists)
    print(min_ind)
    return list(polygons[min_ind])


# print(filter_polygon([[0, 0, 0], [1, 1, 1], [4, 4, 4], [5, 5, 5]], [0, 0, 0]))


def sort_line(polygons):
    """给一个list的多边形，找到一个开头，之后从这个开头开始dfs的顺序给序列排序.
    Parameters
    ----------
    points : type
        Description of parameter `points`.
    dist : type
        Description of parameter `dist`.
    Returns
    -------
    type
        Description of returned object.
    """
    # 从最低点开始
    polygons.sort(key=lambda a: [a.center[2], a.center[1], a.center[0]], reverse=True)
    curr_point = polygons[-1].center
    ordered = []
    while len(polygons) != 0:
        # 找最近的点
        min_dist = dist(curr_point, polygons[-1].center)
        min_ind = len(polygons) - 1
        for ind in range(len([polygons])):
            curr_dist = dist(polygons[ind].center, curr_point)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_ind = ind
        curr_point = polygons[min_ind].center
        ordered.append(polygons[min_ind])
        polygons.pop(min_ind)
    return ordered


# print(sort_line([[0, 0, 0], [0, 0, 4], [0, 0, 3], [0, 0, 2], [0, 0, 6], [0, 1, 5.1], [0, 1, 6], [0, 0, 5]]))


def to_pinyin(name, nonum=False):
    new_name = ""
    for ch in name:
        if u"\u4e00" <= ch <= u"\u9fff":
            new_name += pinyin(ch, style=Style.NORMAL)[0][0]
        else:
            if nonum and ("0" <= ch <= "9" or ch == "_"):
                continue
            new_name += ch
    return new_name
