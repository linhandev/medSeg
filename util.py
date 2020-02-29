'''
包含环境变量和常用函数
'''
import sys
if '/home/aistudio/external-libraries' not in sys.path:
    sys.path.append('/home/aistudio/external-libraries')

import numpy as np
import math
import scipy.ndimage
from config import *
# TODO: 清理函数，去除不需要的，对需要的加上清晰注释


def listdir(path):
    dirs=os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")

    dirs.sort()  #通过一样的sort保持vol和seg的对应
    return dirs

def patch_pos(index,patch_size,stride):
    '''
        index从0开始
    '''
    if len(index)==3:
        x=index[0]
        y=index[1]
        z=index[2]
        return [x*stride[0],y*stride[1],z*stride[2]],[x*stride[0]+patch_size[0],y*stride[1]+patch_size[1],z*stride[2]+patch_size[2]]

    x=index[0]
    y=index[1]
    return [x*stride[0],y*stride[1]],[x*stride[0]+patch_size[0],y*stride[1]+patch_size[1]]


def get_steps(volume_size,patch_size,stride):
    steps=np.asarray(volume_size)
    stride=np.asarray(stride)
    patch_size=np.asarray(patch_size)
    steps=steps+2*stride-patch_size
    steps=np.ceil(steps/stride)
    steps=steps.astype(np.int32)
    return steps
'''
steps=get_steps([18,18],[6,6])
for x in range(0,steps[0]):
    for y in range(0,steps[1]):
            print(patch_pos([x,y],[12,12],[6,6]))
'''
def get_pad(volume_size,patch_size,stride,steps):
    volume_size=np.asarray(volume_size)
    patch_size=np.asarray(patch_size)
    stride=np.asarray(stride)
    full_size=np.multiply(steps,stride)+patch_size
    ul=np.floor((full_size-volume_size)/2)
    lr=full_size-volume_size-ul
    return [ [int(ul[i]),int(lr[i])] for i in range(0,len(ul))]

def get_pleatue(volume_size,pad_len):
    return [[  int(pad_len[i][0]) , int(volume_size[i]-pad_len[i][1])] for i in range(0,len(volume_size))]
'''
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
'''

def crop_pad(volume,pad):
    return volume[
                    pad[0][0]:volume.shape[0]-pad[0][1],
                    pad[1][0]:volume.shape[1]-pad[1][1],
                    pad[2][0]:volume.shape[2]-pad[2][1]
                 ]
'''
print(crop_pad(np.ones([4,3,5]),[[1,1],[0,0],[2,1]]))
'''
def weight_matrix(a,b,size):
    if len(size)==3:
        mat=np.array(
        [
            [
                [a,a,a],
                [a,a,a],
                [a,a,a]
            ],
            [
                [a,a,a],
                [a,b,a],
                [a,a,a]
            ],
            [
                [a,a,a],
                [a,a,a],
                [a,a,a]
            ]
        ]
        )
        weight=scipy.ndimage.interpolation.zoom(mat,[size[0]/3,size[1]/3,size[2]/3],order=1)
        return weight

    mat=np.array(
    [
        [a,a,a],
        [a,b,a],
        [a,a,a]
    ]
    )
    weight=scipy.ndimage.interpolation.zoom(mat,[size[0]/3,size[1]/3],order=1)
    return weight


def get_weight(a,b,size):
    wm=weight_matrix(a,b,[6,6])
    patch=np.ones([6,6])
    result=np.zeros([40,40])
    steps=get_steps([36,36],[6,6],[3,3])

    for x in range(0,steps[0]):
        for y in range(0,steps[1]):
            ul,lr=patch_pos([x,y],[6,6],[3,3])
            respart=np.multiply(patch,wm)
            result[ul[0]:lr[0] , ul[1]:lr[1] ]=result[ul[0]:lr[0] , ul[1]:lr[1] ] + respart
    maxi=result[18][18]
    return weight_matrix(a,b,size),maxi

#print(get_weight(0.2,1,[10,10]))







def dice_coefficent(prediction,label,size,batch_size=1):
    '''
        2 * (x交y) / (|x|+|y|)
    '''
    dice=0
    for batch in range(0,batch_size):
        inter=0
        union=2*size[0]*size[1]*size[2]

        for x in range(0,size[0]):
            for y in range(0,size[1]):
                for z in range(0,size[2]):
                    if(prediction[batch][x][y][z][0]==label[batch][x][y][z][0]):
                        inter=inter+1
        dice=dice+2*inter/union
    return dice/batch_size



'''
prediction=np.ones((2,20,20,20,1))
label=np.ones((2,20,20,20,1))
print(dice_coefficent(prediction,label,[20,20,20]))
'''

'''
def get_non0_volume(volume):


    return [[,],[,],[,]]
'''



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
    bb_min = np.maximum(np.floor(bb_min - bb_size * padding),
                        0).astype('int32')
    bb_max = np.minimum(np.ceil(bb_max + bb_size * padding),
                        volume.shape).astype('int32')

    for i in range(bb_min.shape[0]):
        volumes.append(volume[bb_min[i][0]:bb_max[i][0],
                              bb_min[i][1]:bb_max[i][1],
                              bb_min[i][2]:bb_max[i][2]])
    return volumes


def clip_label(label, category):
    # 有时候标签会包含多种标注，一般是0背景，从1开始随着数变大标记的东西变小
    # label是ndarray，category是最后成为1的类别号,max是最大的类别号
    label[label < category] = 0
    label[label >= category] = 1
    return label


def get_pad_len(volume_shape, pad_size):
    # 获取各个方向应该pad的长度
    margin = []
    for x, y in zip(volume_shape, pad_size):
        if x > y:
            raise Exception("数据的大小比需要pad到的大小更大", volume_shape, pad_size)
        margin.append(y - x)
    margin = [[int(math.floor(x / 2)), y - int(math.floor(x / 2)) - z]
              for x, y, z in zip(margin, pad_size, volume_shape)]
    return margin


def pad_volume(volume, pad_size, pad_value=0):
    # 将volume放在中间，用valuepad到size大小
    pad = pad_size
    if isinstance(pad_size, int):
        pad_size = [pad for i in range(volume.ndim)]
    margin = get_pad_len(volume.shape, pad_size)
    # print(margin)
    # print(type(margin))
    margin[2][0] = 0
    margin[2][1] = 0
    volume = np.pad(volume, margin, 'constant', constant_values=(pad_value))
    # print(volume.shape)
    return volume