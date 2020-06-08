"""
包含环境变量和常用函数
"""
import os
import sys


if "/home/aistudio/external-libraries" not in sys.path:
    sys.path.append("/home/aistudio/external-libraries")

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


def patch_pos(index, patch_size, stride):
    """
		index从0开始
	"""
    if len(index) == 3:
        x = index[0]
        y = index[1]
        z = index[2]
        return (
            [x * stride[0], y * stride[1], z * stride[2]],
            [x * stride[0] + patch_size[0], y * stride[1] + patch_size[1], z * stride[2] + patch_size[2]],
        )

    x = index[0]
    y = index[1]
    return [x * stride[0], y * stride[1]], [x * stride[0] + patch_size[0], y * stride[1] + patch_size[1]]


def get_steps(volume_size, patch_size, stride):
    steps = np.asarray(volume_size)
    stride = np.asarray(stride)
    patch_size = np.asarray(patch_size)
    steps = steps + 2 * stride - patch_size
    steps = np.ceil(steps / stride)
    steps = steps.astype(np.int32)
    return steps


"""
steps=get_steps([18,18],[6,6])
for x in range(0,steps[0]):
	for y in range(0,steps[1]):
			print(patch_pos([x,y],[12,12],[6,6]))
"""


def get_pad(volume_size, patch_size, stride, steps):
    volume_size = np.asarray(volume_size)
    patch_size = np.asarray(patch_size)
    stride = np.asarray(stride)
    full_size = np.multiply(steps, stride) + patch_size
    ul = np.floor((full_size - volume_size) / 2)
    lr = full_size - volume_size - ul
    return [[int(ul[i]), int(lr[i])] for i in range(0, len(ul))]


def get_pleatue(volume_size, pad_len):
    return [[int(pad_len[i][0]), int(volume_size[i] - pad_len[i][1])] for i in range(0, len(volume_size))]


"""
print(get_steps([18,18],[11,11],[6,6]))

volume=np.ones([18,18])
print("shape",volume.shape)
pad_len=get_pad([18,18],[11,11],[6,6],get_steps([18,18],[11,11],[6,6]))
print(pad_len)

volume=np.pad(volume,pad_len,mode="constant")
print(volume.shape)
print(volume)

pleatue=get_pleatue(volume.shape,pad_len)
print(pleatue)

volume=volume[  pleatue[0][0]:pleatue[0][1],  pleatue[1][0]:pleatue[1][1]]
print(volume.shape)
print(volume)
"""


def crop_pad(volume, pad):
    return volume[
        pad[0][0] : volume.shape[0] - pad[0][1], pad[1][0] : volume.shape[1] - pad[1][1], pad[2][0] : volume.shape[2] - pad[2][1]
    ]


"""
print(crop_pad(np.ones([4,3,5]),[[1,1],[0,0],[2,1]]))
"""


def weight_matrix(a, b, size):
    if len(size) == 3:
        mat = np.array([[[a, a, a], [a, a, a], [a, a, a]], [[a, a, a], [a, b, a], [a, a, a]], [[a, a, a], [a, a, a], [a, a, a]]])
        weight = scipy.ndimage.interpolation.zoom(mat, [size[0] / 3, size[1] / 3, size[2] / 3], order=1)
        return weight

    mat = np.array([[a, a, a], [a, b, a], [a, a, a]])
    weight = scipy.ndimage.interpolation.zoom(mat, [size[0] / 3, size[1] / 3], order=1)
    return weight


def get_weight(a, b, size):
    wm = weight_matrix(a, b, [6, 6])
    patch = np.ones([6, 6])
    result = np.zeros([40, 40])
    steps = get_steps([36, 36], [6, 6], [3, 3])

    for x in range(0, steps[0]):
        for y in range(0, steps[1]):
            ul, lr = patch_pos([x, y], [6, 6], [3, 3])
            respart = np.multiply(patch, wm)
            result[ul[0] : lr[0], ul[1] : lr[1]] = result[ul[0] : lr[0], ul[1] : lr[1]] + respart
    maxi = result[18][18]
    return weight_matrix(a, b, size), maxi


# print(get_weight(0.2,1,[10,10]))


def dice_coefficent(prediction, label, size, batch_size=1):
    """
		2 * (x交y) / (|x|+|y|)
	"""
    dice = 0
    for batch in range(0, batch_size):
        inter = 0
        union = 2 * size[0] * size[1] * size[2]

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                for z in range(0, size[2]):
                    if prediction[batch][x][y][z][0] == label[batch][x][y][z][0]:
                        inter = inter + 1
        dice = dice + 2 * inter / union
    return dice / batch_size


"""
prediction=np.ones((2,20,20,20,1))
label=np.ones((2,20,20,20,1))
print(dice_coefficent(prediction,label,[20,20,20]))
"""

"""
def get_non0_volume(volume):


	return [[,],[,],[,]]
"""


def get_bbs(label):
    # 获取一个体中所有为1的区域的bb，返回两个列表，分别是多个病灶最小和最大的下标，最大的是+1的
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
        volumes.append(volume[bb_min[i][0] : bb_max[i][0], bb_min[i][1] : bb_max[i][1], bb_min[i][2] : bb_max[i][2]])
    return volumes


def clip_label(label, category):
    # 有时候标签会包含多种标注，一般是0背景，从1开始随着数变大标记的东西变小
    # label是ndarray，category是最后成为1的类别号,max是最大的类别号
    label[label < category] = 0
    label[label >= category] = 1
    return label


def get_pad_len(volume_shape, pad_size, strict=True):
    """Short summary.

    Parameters
    ----------
    volume_shape : type
        Description of parameter `volume_shape`.
    pad_size : type
        Description of parameter `pad_size`.
    strict : type
        Description of parameter `strict`.

    Returns
    -------
    type
        Description of returned object.

    """

    margin = []
    for x, y in zip(volume_shape, pad_size):
        if y == -1:
            margin.append(0)
            continue
        if x > y:
            if strict:
                raise Exception("Invalid Crop Size", "数据的大小 {} 应小于pad_size {}".format(volume_shape, pad_size))
            else:
                margin.append(0)
            continue
        margin.append(y - x)

    margin = [
        [int(math.floor(x / 2)), y - int(math.floor(x / 2)) - z] if x != 0 else [0, 0]
        for x, y, z in zip(margin, pad_size, volume_shape)
    ]
    return margin


# print(get_pad_len([3, 512, 300], [3, -1, 512]))
# print(get_pad_len([8, 512, 300], [4, -1, 512], False))


def pad_volume(volume, pad_size, pad_value=0, strice=True):
    """将volume放在中间，用 pad_value 填充到 pad_size 大小
    每个维度一共包含三种情况:
    1. 正常: pad_size大于实际大小,那就计算差多少,在这个维度的两侧均匀的补上
    2. 忽略: 不希望改变这个维度的大小,pad_size这个维度填 -1
    3. 错误: volume的大小比 pad_size 还大,在 strice=true 模式下这个报错,终止执行;strice=false模式下这个维度忽略

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
        Description of returned object.

    """
    if isinstance(pad_size, int):
        pad_size = [pad_size for i in range(volume.ndim)]
    margin = get_pad_len(volume.shape, pad_size, strice)
    # print(margin)
    volume = np.pad(volume, margin, "constant", constant_values=(pad_value))
    # print(volume.shape)
    return volume


def filter_largest_volume(vol):
    """ 过滤，只保留最大的volume """
    # TODO 需要能处理 vol 是一个batch的情况
    vol, num = ndimage.label(vol, np.ones([3, 3, 3]))
    maxi = 0
    maxnum = 0
    for i in range(1, num):
        count = vol[vol == i].size
        if count > maxnum:
            maxi = i
            maxnum = count
    vol[vol != maxi] = 0
    vol[vol == maxi] = 1
    return vol


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
    print(header["dim"][1], ",", header["dim"][2], ",", header["dim"][3], end=", ", file=file)
    print(header["pixdim"][1], ",", header["pixdim"][2], ",", header["pixdim"][3], end=", ", file=file)
    print(header["bitpix"], " , ", header["datatype"], file=file)


"""
import nibabel as nib
volf = nib.load('/home/aistudio/data/volume/volume-0.nii')
save_info('v1', volf.header.structarr, 'vol_info.csv')
"""
