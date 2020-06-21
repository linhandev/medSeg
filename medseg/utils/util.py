"""
包含环境变量和常用函数
"""
import os
import sys
import numpy as np
import math
from scipy import ndimage

# TODO: 清理函数，去除不需要的，对需要的加上清晰注释


def listdir(path):
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")
    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs


def save_info(name, header, file_name):
    """这个函数把header里的一些信息写到一个csv里，方便后期看 """
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
        header["pixdim"][1],
        ",",
        header["pixdim"][2],
        ",",
        header["pixdim"][3],
        end=", ",
        file=file,
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
    # 获取一个体中所有为1的区域的bb，返回两个列表，分别是多个前景最小和最大的下标，最大的是+1的
    # TODO: 目前实现了一个病灶，需要实现多个
    one_indexes = np.array(np.where(label == 1))
    if one_indexes.ndim == 0:
        raise Exception("label中没有任何前景")

    bb_min = one_indexes.min(axis=1)
    bb_max = one_indexes.max(axis=1)
    bb_max = bb_max + 1
    return bb_min.reshape(-1, 3), bb_max.reshape(-1, 3)


def crop_to_bbs(volume, bb_min, bb_max, padding=0.3):
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
            volume[
                bb_min[i][0] : bb_max[i][0],
                bb_min[i][1] : bb_max[i][1],
                bb_min[i][2] : bb_max[i][2],
            ]
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


def filter_largest_volume(label):
    """对输入的3D标签进行处理，只保留其中最大的连通块

    Parameters
    ----------
    label : ndarray
        3D array：一个标签
        4D array：一个batch的标签.

    Returns
    -------
    type
        只保留最大连通块的标签.

    """
    is_batch = False if label.ndim == 3 else True
    print(is_batch)
    if label.ndim == 3:
        label = label[np.newaxis, :, :, :]

    for ind in range(label.shape[0]):
        vol, num = ndimage.label(label[ind], np.ones([3, 3, 3]))
        print(vol.dtype)
        print(label[ind].shape)
        print("connected num", num)
        maxi = 0
        maxnum = 0
        for i in range(1, num + 1):
            count = vol[vol == i].size
            print("count", count)
            if count > maxnum:
                maxi = i
                maxnum = count
        print("maxi", maxi)

        vol[vol != maxi] = 0
        vol[vol == maxi] = 1
        label[ind] = vol
    if is_batch:
        return label
    return label[0]


# vol = np.array([[[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]]])
# print(filter_largest_volume(vol))


def save_nii(vol, lab, name="test"):
    import nibabel as nib

    vol = vol.astype("int16")
    volf = nib.Nifti1Image(vol, np.eye(4))
    labf = nib.Nifti1Image(lab, np.eye(4))
    nib.save(volf, "/home/aistudio/data/temp/{}-vol.nii".format(name))
    nib.save(labf, "/home/aistudio/data/temp/{}-lab.nii".format(name))
