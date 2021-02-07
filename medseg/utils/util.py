"""
包含环境变量和常用函数
"""
import os
import sys
import numpy as np
import math
from scipy import ndimage

# from utils.config import cfg

# TODO: 清理函数，去除不需要的，对需要的加上清晰注释


def listdir(path):
    """展示一个路径下所有文件，排序并去除常见辅助文件.

    Parameters
    ----------
    path : type
        Description of parameter `path`.

    Returns
    -------
    type
        Description of returned object.

    """
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs


def save_info(name, header, file_name):
    """将扫描header中的一些信息保存入csv.

    Parameters
    ----------
    name : str
        扫描文件名.
    header : dict
        扫描文件文件头.
    file_name : str
        csv文件的文件名.

    Returns
    -------
    type
        Description of returned object.

    """

    """
	sizeof_hdr      : 348
	data_type       : b''
	db_name         : b''
	extents         : 0
	session_error   : 0
	regular         : b'r'
	dim_info        : 0
	dim             : [  3 512 512  75   1   1   1   1]
	intent_p1       : 0.0
	intent_p2       : 0.0
	intent_p3       : 0.0
	intent_code     : none
	datatype        : int16
	bitpix          : 16
	slice_start     : 0
	pixdim          : [-1.00000e+00  7.03125e-01  7.03125e-01  5.00000e+00  0.00000e+00
	  1.00000e+00  1.00000e+00  5.22410e+04]
	vox_offset      : 0.0
	scl_slope       : nan
	scl_inter       : nan
	slice_end       : 0
	slice_code      : unknown
	xyzt_units      : 10
	cal_max         : 0.0
	cal_min         : 0.0
	slice_duration  : 0.0
	toffset         : 0.0
	glmax           : 255
	glmin           : 0
	descrip         : b'TE=0;sec=52241.0000;name='
	aux_file        : b'!62ABDOMENNATIVUNDVENS'
	qform_code      : scanner
	sform_code      : scanner
	quatern_b       : 0.0
	quatern_c       : 1.0
	quatern_d       : 0.0
	qoffset_x       : 172.9
	qoffset_y       : -179.29688
	qoffset_z       : -368.0
	srow_x          : [ -0.703125   0.         0.       172.9     ]
	srow_y          : [   0.          0.703125    0.       -179.29688 ]
	srow_z          : [   0.    0.    5. -368.]
	intent_name     : b''
	magic           : b'n+1'``
	"""
    file = open(file_name, "a+")
    print(name, end=", ", file=file)
    print(
        header["dim"][1], ",", header["dim"][2], ",", header["dim"][3], end=", ", file=file,
    )
    print(
        header["pixdim"][1], ",", header["pixdim"][2], ",", header["pixdim"][3], end=", ", file=file,
    )
    print(header["bitpix"], " , ", header["datatype"], file=file)
    file.close()


"""
import nibabel as nib
volf = nib.load('/home/aistudio/data/volume/volume-0.nii')
save_info('v1', volf.header.structarr, 'vol_info.csv')
"""


""" 体数据处理 """


def windowlize_image(vol, wwwc):
    """对扫描按照wwwc进行硬crop.

    Parameters
    ----------
    vol : ndarray
        需要进行窗口化的扫描
    ww : int
        窗宽
    wc : int
        窗位

    Returns
    -------
    ndarray
        经过窗口化的扫描
    """
    ww = wwwc[0]
    wc = wwwc[1]
    wl = wc - ww / 2
    wh = wc + ww / 2
    vol = vol.clip(wl, wh)
    return vol


def clip_label(label, category):
    # 有时候标签会包含多种标注，一般是0背景，从1开始随着数变大标记的东西变小
    # label是ndarray，category是最后成为1的类别号,max是最大的类别号
    label[label < category] = 0
    label[label >= category] = 1
    return label


def get_bbs(label):
    """求一个标签中所有前景的bb.

    Parameters
    ----------
    label : ndarray
        标签.

    Returns
    -------
    list
        list中每一个前景区域一个[bb_min, bb_max]，分别是这个前景块bb低和高两个角的坐标.

    """
    # TODO: 目前实现了一个病灶，需要实现多个
    one_indexes = np.array(np.where(label == 1))
    if one_indexes.ndim == 0:
        raise Exception("label中没有任何前景")

    bb_min = one_indexes.min(axis=1)
    bb_max = one_indexes.max(axis=1)
    bb_max = bb_max + 1
    return bb_min.reshape(-1, 3), bb_max.reshape(-1, 3)


def crop_to_bbs(volume, bbs, padding=0.3):
    """将一个扫描的背景mute掉，只留下前景及其周围的区域，支持多个前景块.
    具体做法是创建一个mask，对于bbs中的每个前景块，计算中心位置，按照padding计算保留的块范围(不会超出volume)，在mask中设成1。所有块都计算完之后mute掉mask中还是0的所有位置
    Parameters
    ----------
    volume : ndarray
        扫描.
    bbs : list
        [[bb_min, bb_max], [bb_min, bb_max], ...].
    padding :
        Description of parameter `padding`.

    Returns
    -------
    type
        Description of returned object.

    """

    # 将一个体切成一个或者多个包含1的区域的bb
    # padding 值是在各个维度上向大和小分别拓展多大的视野，一个数就是都一样，列表可以让不同维度不一样
    pd = padding
    if isinstance(padding, float):
        padding = []
        for i in range(volume.ndim):
            padding.append(pd)

    volumes = []
    bb_size = bb_max - bb_min
    bb_min = np.maximum(np.floor(bb_min - bb_size * padding), 0).astype("int32")
    bb_max = np.minimum(np.ceil(bb_max + bb_size * padding), volume.shape).astype("int32")

    for i in range(bb_min.shape[0]):
        volumes.append(
            volume[bb_min[i][0] : bb_max[i][0], bb_min[i][1] : bb_max[i][1], bb_min[i][2] : bb_max[i][2],]
        )
    return volumes


def get_pad_len(volume_shape, pad_size, strict=True):
    # 1. 计算每个维度当前长度和目标差多少
    margin = []
    for x, y in zip(volume_shape, pad_size):
        # 1.1 如果目标 -1 ，那这个维度过
        if y == -1:
            margin.append(0)
            continue
        # 1.2 如果当前长度比目标长度还大，报错或者过
        if x > y:
            if strict:
                raise Exception(
                    "Invalid Crop Size", "数据的大小 {} 应小于 pad_size {}".format(volume_shape, pad_size),
                )
            else:
                margin.append(0)
            continue
        # 1.3 如果正常，目标大于当前维度，做差
        margin.append(y - x)
    # 2. 计算每个维度应该补多少
    res = []
    for m, p, v in zip(margin, pad_size, volume_shape):
        if m == 0:
            # 2.1 margin = 0的略过
            res.append([0, 0])
        else:
            # 2.2  margin分成两份
            half = math.floor(m / 2)
            res.append([half, p - v - half])

    return res


# print(get_pad_len([3, 512, 300], [3, -1, 512]))
# print(get_pad_len([512, 512, 3], [512, 512, 3]))
# print(get_pad_len([8, 512, 300], [4, -1, 512]))
# print(get_pad_len([8, 512, 300], [4, -1, 512], False))


def pad_volume(volume, pad_size, pad_value=0, strice=True):
    """将volume放在中间，用 pad_value 填充到 pad_size 大小
    每个维度一共包含三种情况:
    1. 正常: pad_size大于实际大小，那就计算差多少,在这个维度的两侧均匀的补上
    2. 忽略: 不希望改变这个维度的大小，pad_size这个维度填 -1
    3. 错误: volume的大小比 pad_size 还大，在 strice=true 模式下这个报错,终止执行；strice=false模式下这个维度忽略

    Parameters
    ----------
    volume : type
        Description of parameter `volume`.
    pad_size : int/list/tuple
        如果是一个int,那么做成一个和volume.ndim维的list,三个维度的大小一样,按照这个pad;如果是list,tuple直接按照这个pad
    pad_value : type
        Description of parameter `pad_value`.
    strice : type
        Description of parameter `strice`.

    Returns
    -------
    type
        经过pad的数据

    """
    if isinstance(pad_size, int):
        pad_size = [pad_size for i in range(volume.ndim)]
    margin = get_pad_len(volume.shape, pad_size, strice)
    # print(margin)
    volume = np.pad(volume, margin, "constant", constant_values=(pad_value))
    # print(volume.shape)
    return volume


def filter_largest_bb(label, ratio=1.2):
    """求最大的连通块bb范围，去掉范围外的所有fp。比只保留最大的连通块更保守.

    Parameters
    ----------
    label : ndarray
        分割标签，前景为1.
    ratio: float
        最终bb范围是最大联通块bb范围的多少倍，比如1.2相当于周围有0.1的拓展

    Returns
    -------
    type
        经过处理的标签.

    """
    # 求最大连通块
    vol, num = ndimage.label(label, np.ones([3 for _ in range(label.ndim)]))
    maxi = 0
    maxnum = 0
    for i in range(1, num + 1):
        count = vol[vol == i].size
        if count > maxnum:
            maxi = i
            maxnum = count
    maxind = np.where(vol == maxi)
    # 求最大连通块的bb范围
    ind_range = [[np.min(maxind[axis]), np.max(maxind[axis]) + 1] for axis in range(label.ndim)]
    ind_len = [r[1] - r[0] for r in ind_range]
    ext_ratio = (ratio - 1) / 2
    # 求加上拓展的边缘
    clip_range = [[r[0] - int(l * ext_ratio), r[1] + int(l * ext_ratio)] for r, l in zip(ind_range, ind_len)]
    for ind in range(len(clip_range)):
        if clip_range[ind][0] < 0:
            clip_range[ind][0] = 0
        if clip_range[ind][1] > label.shape[ind]:
            clip_range[ind][1] = label.shape[ind]
    r = clip_range
    # print(r)
    # 去掉拓展外的fp
    new_lab = np.zeros(label.shape)
    # 内部所有前景都保留
    new_lab[r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]] = label[
        r[0][0] : r[0][1], r[1][0] : r[1][1], r[2][0] : r[2][1]
    ]
    return new_lab


# filter_largest_bb(
#     np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), 3
# )


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
    vol, num = ndimage.label(label, np.ones([3, 3, 3]))
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


# vol = np.array([[[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]]])
# vol = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]])
# print(filter_largest_volume(vol, 3, mode="soft"))


def save_nii(vol, lab, name="test"):
    import nibabel as nib

    vol = vol.astype("int16")
    volf = nib.Nifti1Image(vol, np.eye(4))
    labf = nib.Nifti1Image(lab, np.eye(4))
    nib.save(volf, "/home/aistudio/data/temp/{}-vol.nii".format(name))
    nib.save(labf, "/home/aistudio/data/temp/{}-lab.nii".format(name))


def slice_count():
    """数所有的npz总共包含多少slice.

    Returns
    -------
    int
        所有npz中slice总数.

    """
    tot = 0
    npz_names = listdir(cfg.TRAIN.DATA_PATH)
    for npz_name in npz_names:
        data = np.load(os.path.join(cfg.TRAIN.DATA_PATH, npz_name))
        lab = data["labs"]
        tot += lab.shape[0]
    return tot


# print(slice_count())


def cal_direction(fname, scan, label):
    """根据预存信息矫正患者体位.

    Parameters
    ----------
    fname : str
        患者文件名.
    scan : ndarray
        3D扫描.
    label : type
        3D标签.

    Returns
    -------
    ndarray, ndarray
        校准后的3D数组.

    """
    f = open("./config/directions.csv")
    dirs = f.readlines()
    f.close()
    # print("dirs: ", dirs)
    dirs = [x.rstrip("\n") for x in dirs]
    dirs = [x.split(",") for x in dirs]
    dic = {}
    for dir in dirs:
        dic[dir[0].strip()] = dir[1].strip()
    dirs = dic
    try:
        if dirs[fname] == "2":
            scan = np.rot90(scan, 3)
            label = np.rot90(label, 3)
        else:
            scan = np.rot90(scan, 1)
            label = np.rot90(label, 1)
    except KeyError:
        pass
    return scan, label
