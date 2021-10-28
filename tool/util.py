import concurrent.futures
import logging
import math
import multiprocessing
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import trimesh
from pypinyin import Style, pinyin
from tqdm import tqdm

logging.basicConfig(level=logging.NOTSET)


def listdir(path, sort=True):
    """获取一个路径下所有文件名，去除常见辅助文件，排序

    Parameters
    ----------
    path : str
        目标路径
    sort : bool
        是否排序

    Returns
    -------
    list
        所有文件名

    """
    names = os.listdir(path)
    skips = [".DS_Store"]
    for skip in skips:
        if skip in names:
            names.remove(skip)
    if sort:
        names.sort()
    return names


def blood_sort(polygons):
    """按照血流反向对血管中的圆进行排序.

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
            min_height = p.height
            min_ind = idx

    # 2. 一直找无序的list中和当前最后一个距离最近的作为下一个
    ordered.append(polygons[min_ind])
    del polygons[min_ind]

    while len(polygons) != 0:
        min_dist = dist(ordered[-1].center, polygons[0].center)
        min_ind = 0
        for idx, p in enumerate(polygons):
            d = dist(ordered[-1].center, p.center)
            if d < min_dist:
                min_dist = d
                min_ind = idx
        ordered.append(polygons[min_ind])
        del polygons[min_ind]
    for idx, p in enumerate(ordered):
        p.idx = idx
    return ordered


def filter_largest_volume(label, ratio=1.2, mode="soft"):
    """对输入的一个3D标签进行处理，只保留其中最大的连通块

    Parameters
    ----------
    label : ndarray
        3D array：一个分割标签
    ratio : float
        分割保留的范围
    mode : str
        "soft" / "hard"
        hard是只保留最大的联通块，soft是保留最大连通块bb内的

    Returns
    -------
    type
        只保留最大连通块的标签.

    """
    if mode == "soft":
        return filter_largest_bb(label, ratio)
    vol, num = scipy.ndimage.label(label, np.ones([3, 3, 3]))
    maxi = 0
    maxnum = 0
    for i in range(1, num + 1):
        count = vol[vol == i].size
        if count > maxnum:
            maxi = i
            maxnum = count

    vol[vol != maxi] = 0
    vol[vol == maxi] = 1
    label = vol
    return label


def toint(x):
    return [int(num) for num in x]


def slice_med(
    scan_path,
    scan_img_dir,
    label_path=None,
    label_img_dir=None,
    thich=3,
    rot=0,
    wwwc=(1000, 0),
    thresh=None,
    front=None,
    front_mode=None,
    itv=1,
    resize=None,
    ext="png",
    transpose=False,
    prefix="",
):
    """将扫描和标签转成2D切片.
    扫描和标签一起处理，支持窗口化，旋转，略过没有前景的片，隔固定数量的片取一片

    Parameters
    ----------
    scan_path : str
        扫描nii路径.
    scan_img_dir : str
        扫描生成png放到这.
    label_path : str
        标签nii路径.
    label_img_dir : str
        标签生成png放到这.
    thick : int
        保存切片的厚度，默认3层
    rot : int
        进行几次旋转，如果有标签会一起.
    wwwc : list/tuple
        进行窗口化的窗宽窗位.
    thresh : int
        标签中前景数量达到这个数才生成png，否则略过.
    front: int
        标签中只保留0和front两个值
    front_mode: str
        - stack:比front小的变成0，比front大的变成front。适合脏器+内部肿瘤这种包含的情况
        - single:只保留front，其他的都变成0。适合多种脏器的情况
        - None:不进行处理
    itv: int
        每隔itv层取1层
    resize： list
        对切片resize到的大小，None是不进行resize，否则给一个列表两个数字，比如[512, 512]
    ext: str
        结果保存的格式
        - 图片：cv2.imwrite 支持的图片格式都可以，推荐png。不带点！保存成灰度图会把数据范围拉到0～255
        - npy：保存成npy格式
    transpose: bool
        是否调整维度顺序
    prefix: str
        输出文件前缀
    """
    # 1. 读取扫描和标签
    # 格式应为 [层，横向，竖向]
    print(scan_path, label_path)
    scanf = nib.load(scan_path)
    scan_data = scanf.get_fdata()
    if transpose:
        scan_data = np.transpose(scan_data, [2, 0, 1])
    name = osp.basename(scan_path)
    # print(scan_data.shape)

    # 2. 进行窗宽窗位处理 (早做少吃内存)
    wl, wh = (wwwc[1] - wwwc[0] / 2, wwwc[1] + wwwc[0] / 2)
    scan_data = scan_data.astype("float32").clip(wl, wh)
    if ext == "npy":
        scan_data = scan_data.astype("uint16")
    else:
        scan_data = (scan_data - wl) / (wh - wl) * 255
        scan_data = scan_data.astype("uint8")

    if label_path:
        # labelf = sitk.ReadImage(label_path)
        # label_data = sitk.GetArrayFromImage(labelf)
        labelf = nib.load(label_path)
        label_data = labelf.get_fdata()
        if transpose:
            label_data = np.transpose(label_data, [2, 0, 1])
        # 1.1 有多种目标的标签保留一个前景
        if front_mode:
            if front_mode == "stack":
                label_data[label_data < front] = 0
                label_data[label_data >= front] = 1
            if front_mode == "single":
                label_data[label_data != front] = 0
                label_data[label_data != 0] = 1
        if label_data.shape != scan_data.shape:
            logging.error(
                f"[ERROR] Patient {name}'s scan and image dimension mismatch, scan shape is {scan_data.shape}, label shape is {label_data.shape}"
            )
            return
        label_data = label_data.astype("uint8")

    # 3. 对大小不对的进行resize
    if resize and scan_data.shape[1:] != resize:
        # TODO: 对标签和图像进行插值
        print("need resize")
        # vol = scipy.ndimage.interpolation.zoom(vol, [0.5, 0.5, 1], order=1 if islabel else 3)

    # 4. 旋转图像
    scan_data = np.rot90(scan_data, rot, (1, 2))
    if label_path:
        label_data = np.rot90(label_data, rot, (1, 2))

    # 5. 复制第一层和最后一层，避免多层的切片少最前和最后的几层
    gap = int((thich - 1) / 2)

    for _ in range(gap):
        scan_data = np.concatenate(
            [
                scan_data[0][np.newaxis, :, :],
                scan_data,
                scan_data[-1][np.newaxis, :, :],
            ],
            axis=0,
        )
        if label_path:
            label_data = np.concatenate(
                [
                    label_data[0][np.newaxis, :, :],
                    label_data,
                    label_data[-1][np.newaxis, :, :],
                ],
                axis=0,
            )

    # 6. 准备路径，进行切片和存盘
    if not os.path.exists(scan_img_dir):
        os.makedirs(scan_img_dir)
    if label_path and not os.path.exists(label_img_dir):
        os.makedirs(label_img_dir)
    # 计算zfill长度
    fill_len = 1
    while scan_data.shape[0] > 10 ** fill_len:
        fill_len += 1
    fill_len += 1
    name = name.split(".")[0]
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    )
    for ind in range(gap, scan_data.shape[0] - gap, itv):  # TODO: 支持任意层厚
        # 6.1 可能标签中前景过少触发跳过，所以先处理标签
        if label_path:
            label_slice = label_data[ind, :, :]
            # 前景不到thresh就跳过
            if thresh is not None and label_slice.sum() <= thresh:
                continue
            file_path = osp.join(
                label_img_dir,
                f"{prefix}{name}-{str(ind - 1).zfill(fill_len)}.{ext}",
            )
            executor.submit(save_slice, label_slice, file_path)
            # save_slice(label_slice, file_path)
        scan_slice = scan_data[ind - gap : ind + gap + 1, :, :]
        file_path = osp.join(
            scan_img_dir,
            f"{prefix}{name}-{str(ind-1).zfill(fill_len)}.{ext}",
        )
        executor.submit(save_slice, scan_slice, file_path)
        # save_slice(scan_slice, file_path)


def save_slice(slice, file_path):
    """保存一个切片

    Parameters
    ----------
    slice : numpy.ndarray
        保存的切片数据，CWH格式
    file_path : str
        保存文件路径
    """
    # 1. 调整文件格式到WHC
    if len(slice.shape) == 2:
        slice = slice[:, :, np.newaxis]
    else:
        slice = np.swapaxes(slice, 0, 2)
        slice = np.swapaxes(slice, 0, 1)
    # 2. 按照文件路径确定保存格式，写盘
    if file_path.endswith("npy"):
        pass
    else:
        cv2.imwrite(file_path, slice)


def check_nii_match(scan_dir, label_dir, skip=["", ""]):
    """检查两个目录下的扫描和标签是不是对的上
    只支持nii或nii.gz文件

    Parameters
    ----------
    scan_dir : str
        扫描所在路径.
    label_dir : str
        标签所在路径.
    skip : list
        在检查扫描和标签配对的时候，扫描和标签分别lstrip skip[0] 和 skip[1]，用于配对 scan-1.nii 和 label-1.nii 这种情况

    Returns
    -------
    bool
        比较的结果，对上了返回True，否则False，具体的细节直接打到logging.
    """
    # 1. 获取两个路径下的文件，过滤，去拓展名
    pass_check = True
    scans = listdir(scan_dir)
    labels = listdir(label_dir)
    scans = [n for n in scans if n.endswith("nii") or n.endswith("gz")]
    labels = [n for n in labels if n.endswith("nii") or n.endswith("gz")]

    scan_names = [n.split(".")[0].lstrip(skip[0]) for n in scans]
    label_names = [n.split(".")[0].lstrip(skip[1]) for n in labels]

    # 2. 检查图像和扫描文件数量和文件名能不能对上
    if len(scan_names) != len(label_names):
        logging.error(
            f"Number of scans ({len(scan_names)}) and labels ({len(label_names)}) don't match"
        )
        pass_check = False
    else:
        logging.info("Scan and label file number matches!")

    names_match = True
    scan_set, label_set = set(scan_names), set(label_names)
    scan_without_label = scan_set - label_set
    label_without_scan = label_set - scan_set
    intersect = scan_set.intersection(label_set)
    if len(scan_without_label) != 0:
        names_match = False
        logging.error(
            f"Following scans don't have corresponding label:\n {(' ' + skip[0]).join(scan_without_label)}"
        )

    if len(label_without_scan) != 0:
        Number
        names_match = False
        logging.error(
            f"Following labels don't have corresponding scans:\n {(' ' + skip[1]).join(scan_without_label)}"
        )

    if names_match:
        logging.info("All file names matche")
    else:
        pass_check = False

    # 3. 检查有扫描和标签的pair：header有没有问题，扫描和标签大小能不能对上
    for name in intersect:
        # 3.1 拓展名
        if skip[0] + name + ".nii" in scans:
            scan = skip[0] + name + ".nii"
        if skip[0] + name + ".nii.gz" in scans:
            scan = skip[0] + name + ".nii.gz"
        if skip[1] + name + ".nii" in labels:
            label = skip[1] + name + ".nii"
        if skip[1] + name + ".nii.gz" in labels:
            label = skip[1] + name + ".nii.gz"

        scanf = nib.load(os.path.join(scan_dir, scan))
        labelf = nib.load(os.path.join(label_dir, label))
        if (scanf.affine == np.eye(4)).all():
            logging.warn(f"Scan {scan} have np.eye(4) affine, check the header")
        if (labelf.affine == np.eye(4)).all():
            logging.warn(f"Label {label} have np.eye(4) affine, check the header")
        if not (labelf.header["dim"] == scanf.header["dim"]).all():
            logging.error(
                f"Label and scan dimension mismatch for {scan} and {label}, check_scan is {scanf.header['dim'][1:4]}, label is {labelf.header['dim'][1:4]}"
            )
            pass_check = False
    return pass_check


def read_slice(file_path):
    if file_path.endswith("npy"):
        pass
    else:
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def inspect_slice(scan_path, label_path=None, wwwc=[1000, 0]):
    """可视化2D的切片
    scan_path 的文件按照wwwc做窗宽窗位。scan_path 也可以传分割标签，wwwc对应写也可以看
    Parameters
    ----------
    scan_path : str
        扫描或标签文件路径，支持图片和npy格式。多层的扫描会展示多个图像
    label_path : str
        标签路径，会叠加到scan的中间一层展示
    wwwc : list/tuple
        窗宽窗位，会apply到scan上
    """

    scan = read_slice(scan_path)
    if label_path:
        label = read_slice(label_path)

    print(scan.shape, label.shape)
    # TODO: 对多片的scan分成多片展示
    # TODO: label叠加到scan上展示
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 2, 1)
    plt.imshow(scan)
    fig.add_subplot(1, 2, 2)
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
        self.center = list(
            (np.min(self.points, axis=0) + np.max(self.points, axis=0)) / 2
        )
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
            return math.atan(
                (a[1] - self.base[1]) / (a[0] - self.base[0] + self.epsilon)
            )

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
            unit = (
                y / ((x ** 2 + y ** 2) ** (1 / 2) + epsilon),
                -x / ((x ** 2 + y ** 2) ** (1 / 2) + epsilon),
                0,
            )
            matrix = trimesh.transformations.rotation_matrix(-angle, unit, (0, 0, 0))
            p.append(1)
            p = np.array(p).reshape([1, 4])
            p = p.transpose()
            res = np.dot(matrix, p)
            return [float(res[0]), float(res[1])]

        self.points = [
            [p[0] - self.base[0], p[1] - self.base[1], p[2] - self.base[2]]
            for p in self.points
        ]
        # print("+_+", self.center)
        self.center = [b - a for a, b in zip(self.base, self.center)]
        self.center = rot_to_horizontal(self.center)
        # print("_+_", self.center)
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
        for alpha in np.arange(
            ang_range[0], ang_range[1], (ang_range[1] - ang_range[0]) / split
        ):
            # TODO: 如果这个线是垂直的
            if alpha == np.pi / 2:
                continue
            k = math.tan(alpha)
            # print(alpha)

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
# print(po.cal_diameter(split=2, step=0.1))


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
            if dist(curr_point.center, points[ind].center) < thresh:
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
        if "\u4e00" <= ch <= "\u9fff":
            new_name += pinyin(ch, style=Style.NORMAL)[0][0]
        else:
            # if nonum and ("0" <= ch <= "9" or ch == "_"):
            #     continue
            new_name += ch
    return new_name
