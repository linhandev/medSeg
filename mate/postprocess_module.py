#-*- coding:utf-8 -*-

import cv2
import numpy as np
from skimage.measure import label as label_function
from skimage.measure import regionprops
from skimage import morphology
from lib.error import Error, ErrorCode
from lib.info_dict_module import InfoDict
from lib.anti_interprolation_module import anti_interp_3d_zyx
from lib.large_image_checker_module import anti_check_large_image_zyx
from lib.rotate_module import rotate_label_3D


def im_train_anticrop(imgs, four_corner, size_interpolate):

    label_new = np.zeros(size_interpolate)
    for i, imgs_slice in enumerate(imgs):
        label_new[i, four_corner[0]:four_corner[1], four_corner[2]:four_corner[3]] = imgs_slice
    return label_new

def conneted_area_2D(pre_2D):
    # 每层保留最大两个连通域
    pre_2D = np.array(pre_2D,np.uint8)
    pre_2D = cv2.GaussianBlur(pre_2D, (3, 3), 1)
    for idx, pre in enumerate(pre_2D):
        pre_2D[idx, :, :] = morphology.remove_small_holes(pre_2D[idx, :, :] > 0, min_size=1000, in_place=False)
        pre_2D[idx, :, :] = morphology.remove_small_objects(pre_2D[idx, :, :] > 0, min_size=35, in_place=False)

        if np.max(pre_2D[idx, :, :]) > 0:
            pre_2D[idx, :, :] = morphology.dilation(pre_2D[idx, :, :], morphology.disk(2))
            pre_2D[idx, :, :] = label_function(pre_2D[idx, :, :])
            return_property = regionprops(pre_2D[idx, :, :])
            return_ordered = sorted(return_property, key=lambda s: s.area, reverse=True)
            if len(return_ordered) >= 2 and (return_ordered[1].area > 1):
                pre_2D[idx, :, :] = morphology.remove_small_objects(pre_2D[idx, :, :] > 0,min_size=int(return_ordered[1].area - 1),connectivity=1, in_place=True)
                pre_2D[idx, :, :] = morphology.erosion(pre_2D[idx, :, :], morphology.disk(3))
            else:
                pre_2D[idx, :, :] = morphology.remove_small_objects(pre_2D[idx, :, :] > 0,
                                                                    min_size=int(return_ordered[0].area - 1),
                                                                    connectivity=1, in_place=True)
                pre_2D[idx, :, :] = morphology.erosion(pre_2D[idx, :, :], morphology.disk(3))
    pre_2D[pre_2D > 0] = 1
    pre_2D = np.array(pre_2D, np.uint8)

    return pre_2D

def conneted_area_3D(pre_2D):
    pre_2D = np.array(pre_2D,np.uint8)
    if np.max(pre_2D) > 0:
        pre_2D_label = label_function(pre_2D)
        return_property = regionprops(pre_2D_label)
        return_ordered = sorted(return_property, key=lambda s: s.area, reverse=True)
        if len(return_ordered) > 1 and (return_ordered[0].area > 1):
            pre_2D = morphology.remove_small_objects(pre_2D_label, min_size=int(return_ordered[0].area - 1),connectivity=1, in_place=True)

        pre_2D[pre_2D > 0] = 1
        pre_2D = np.array(pre_2D, np.uint8)

    return pre_2D

def GaussianBlur_3D(pre_2D, ksize, sigmaX):
    idx = []
    for i, label_slice in enumerate(pre_2D):
        if np.max(label_slice) > 0:
            idx.append(i)
    if idx:
        pre_2D[idx, :, :] = cv2.GaussianBlur(pre_2D[idx, :, :], ksize, sigmaX)

    return pre_2D



def remove_small_label(pre_2D, min_size, redetermine = False):
    """
    去除小objects， 因为spacing归到1，所以此min_size有意义
    redetermine指重新确定，当以min_size为大小去除后，整个label为空，则减小min_size为原来的一半，再次对原数据进行去除，
    如果整个label仍为空，则不进行去除操作

    # Arguments
        pre_2D: 预测后已经剪裁过的label
        min_size: 删除最小objects的点数
        redetermine: 是否进行重新判定
    # Returns
        pre_2D: 去除小label后的预测矩阵

    # Example
    """
    # 去除小objects， 因为spacing归到1，所以此min_size有意义
    if redetermine:
        tmp_pre_2D = np.zeros(pre_2D.shape)
        for i, pre in enumerate(pre_2D):
            if np.max(pre) > 0:
                # print(np.count_nonzero(pre))
                tmp_pre_2D[i, :, :] = morphology.remove_small_objects(pre > 0, min_size=min_size)

        if np.max(np.max(tmp_pre_2D)) > 0:
            pre_2D = tmp_pre_2D
        else:
            tmp_pre_2D = np.zeros(pre_2D.shape)
            for i, pre in enumerate(pre_2D):
                if np.max(pre) > 0:
                    # print(np.count_nonzero(pre))
                    tmp_pre_2D[i, :, :] = morphology.remove_small_objects(pre > 0, min_size=int(min_size/2))

            if np.max(np.max(tmp_pre_2D)) > 0:
                pre_2D = tmp_pre_2D

    else:
        for i, pre in enumerate(pre_2D):
            if np.max(pre) > 0:
                # print(np.count_nonzero(pre))
                pre_2D[i, :, :] = morphology.remove_small_objects(pre > 0, min_size=min_size)

    return pre_2D

def crop_by_size_and_shift(imgs, image_size, center=None, pixely=0, pixelx=0):
    '''
    剪切函数，相当于原来的cutting()，pixelx、pixely的移动值是相对图像中心
    注意参数次序和原来的cutting函数不同，shift参数不用传入
    :param imgs: 需要裁剪的数据
    :param image_size: 需要的图像大小
    :param pixely: 偏移y
    :param pixelx: 偏移x
    :return:
    '''

    if len(imgs.shape) == 2:  # 2D image
        imgs = imgs.copy()
        image_sizeY = image_size[0]
        image_sizeX = image_size[1]

        if center is None:
            center = [imgs.shape[0] // 2, imgs.shape[1] // 2]

        pixely = int(center[0] - imgs.shape[0] // 2) + pixely
        pixelx = int(center[1] - imgs.shape[1] // 2) + pixelx

        #    z, x, y = np.shape(imgs)
        y, x = np.shape(imgs)
        shift = np.max([abs(pixely), abs(pixelx), np.max((abs(y - image_sizeY), abs(x - image_sizeX)))])
        judge = sum([y > (image_sizeY + abs(pixely) * 2), x > (image_sizeX + abs(pixelx) * 2)])
        imgs_new = []
        image_std = imgs
        #    for i, image_std in enumerate(imgs):
        if judge == 2:
            image_std = image_std[int((y - image_sizeY) / 2 + pixely):int((y + image_sizeY) / 2 + pixely),
                        int((x - image_sizeX) / 2 + pixelx):int((x + image_sizeX) / 2) + pixelx]
        #        imgs_new.append(image_std)
        else:
            image_new = np.min(image_std) * np.ones([image_sizeY + shift * 2, image_sizeX + shift * 2], dtype=np.int32)
            image_new[int((image_sizeY + shift * 2 - y) / 2):int((image_sizeY + shift * 2 - y) / 2) + y,
            int((image_sizeX + shift * 2 - x) / 2):int((image_sizeX + shift * 2 - x) / 2) + x] = image_std
            y1, x1 = np.shape(image_new)
            image_std = image_new[int((y1 - image_sizeY) / 2 + pixely):int((y1 + image_sizeY) / 2 + pixely),
                        int((x1 - image_sizeX) / 2 + pixelx):int((x1 + image_sizeX) / 2) + pixelx]

        #    imgs_new = np.array(imgs_new, np.float32)
        imgs_new = image_std

    elif len(imgs.shape) == 3:  # 3D image
        imgs = imgs.copy()
        image_sizeY = image_size[0]
        image_sizeX = image_size[1]

        if center is None:
            center = [imgs.shape[1] // 2, imgs.shape[2] // 2]

        pixely = int(center[0] - imgs.shape[1] // 2) + pixely
        pixelx = int(center[1] - imgs.shape[2] // 2) + pixelx

        z, y, x = np.shape(imgs)
        #        x, y = np.shape(imgs)
        shift = np.max([abs(pixely), abs(pixelx), np.max((abs(y - image_sizeY), abs(x - image_sizeX)))])
        judge = sum([y > (image_sizeY + abs(pixely) * 2), x > (image_sizeX + abs(pixelx) * 2)])
        imgs_new = []
        image_std = imgs
        if judge == 2:
            for i, image_std in enumerate(imgs):
                image_std = image_std[int((y - image_sizeY) / 2 + pixely):int((y + image_sizeY) / 2 + pixely),
                            int((x - image_sizeX) / 2 + pixelx):int((x + image_sizeX) / 2) + pixelx]
                imgs_new.append(image_std)
        else:
            for i, image_std in enumerate(imgs):
                # 按最小值填补imgs外不足部分
                image_new = np.min(image_std) * np.ones([image_sizeY + shift * 2, image_sizeX + shift * 2],
                                                        dtype=np.int32)
                image_new[int((image_sizeY + shift * 2 - y) / 2):int((image_sizeY + shift * 2 - y) / 2) + y,
                int((image_sizeX + shift * 2 - x) / 2):int((image_sizeX + shift * 2 - x) / 2) + x] = image_std
                y1, x1 = np.shape(image_new)
                image_std = image_new[int((y1 - image_sizeY) / 2 + pixely):int((y1 + image_sizeY) / 2 + pixely),
                            int((x1 - image_sizeX) / 2 + pixelx):int((x1 + image_sizeX) / 2) + pixelx]
                imgs_new.append(image_std)

        imgs_new = np.array(imgs_new)
    else:
        Error.exit(ErrorCode.process_input_shape_error)

    return imgs_new

def refill_by_size_and_shift(label_array: np.ndarray, image_raw_size, center, pixely=0, pixelx=0):
    """
    根据图像的大小以及pixelx,pixely的偏移量来填充
    整体流程：
    1. 调用crop_by_size_and_shift
    """
    # 断言保证

    if center is None:
        center = [image_raw_size[0] // 2, image_raw_size[1] // 2]

    pixely = int(center[0] - image_raw_size[0] // 2) + pixely
    pixelx = int(center[1] - image_raw_size[1] // 2) + pixelx

    # 1. 调用crop_by_size_and_shift
    if len(label_array.shape) == 3:

        rst_array = crop_by_size_and_shift(label_array, image_raw_size, None, -pixely, -pixelx)
    elif len(label_array.shape) == 4:
        rst_array = np.zeros((label_array.shape[0], image_raw_size[0], image_raw_size[1], label_array.shape[3]),
                             np.float32)
        for i in range(label_array.shape[3]):
            rst_array[:, :, :, i]= crop_by_size_and_shift(label_array[:, :, :, i], image_raw_size,
                                                                           None, -pixely, -pixelx)

    return rst_array

def pre_recover(pre_2D, interpolation, thresh, info_dict: InfoDict):
    """
    反剪裁，恢复到原CT尺寸

    # Arguments
        pre_2D: 预测后已经剪裁过的label
        interpolation: 插值方法
        thresh: 插值后保留大于thresh的label
        info_dict: 字典
    # Returns
        new_labels: 恢复到原CT尺寸的label

    # Example
    """
    # 反剪裁
    # new_labels = im_train_anticrop(pre_2D, info_dict.four_corner, info_dict.size_2D)
    new_labels = refill_by_size_and_shift(pre_2D, info_dict.size_2D[-2:], info_dict.img_centre)

    # 反插值
    rows_orig_imgs = info_dict.image_shape_raw[1]
    cols_orig_imgs = info_dict.image_shape_raw[2]
    # 对于spacing处于[0.8,1.2）之间的不进行归1，除此以外均插值到1.0
    original_space = info_dict.spacing_list[1]
    if (original_space <= 0.8 or original_space > 1.2):
        # new_labels = image_interp(new_labels, [rows_orig_imgs, cols_orig_imgs], interpolation)
        new_labels = anti_interp_3d_zyx(new_labels, info_dict.spacing_list[1:], info_dict.target_spacing, ori_shape = (rows_orig_imgs, cols_orig_imgs), kind=interpolation)
        new_labels = np.where(new_labels > thresh, 1, 0)

    new_labels = np.array(new_labels, np.uint8)
    return new_labels

def postprocess(pre_2D, info_dict: InfoDict):

    """后处理程序
    # Arguments：
        imgs：原图
        pre：预测后label
        loc_start：分类号的第一张的位置
        num_z, num_rows, num_cols：预处理后裁剪的size
        original_space：原图的spacing
        target_space：插值目标的spacing
        img_centre：身体中心点
        para: 是否保存中间变量 （默认情况False）
    # Returns
        label_stem：后处理完的label

    # Example
    """
    print('\nBegin to postprocess')

    pre_labels = pre_2D[np.newaxis, :, :]

    pre_2D = np.array(pre_labels, np.uint8)

    pre_2D_l = pre_2D.copy()
    pre_2D_l[pre_2D_l != 1] = 0
    pre_2D_r = pre_2D.copy()
    pre_2D_r[pre_2D_r != 2] = 0

    # 反剪裁、反插值
    pre_2D_l = pre_recover(pre_2D_l, cv2.INTER_LINEAR, 0, info_dict)
    pre_2D_r = pre_recover(pre_2D_r, cv2.INTER_LINEAR, 0, info_dict)


    pre_2D_r[pre_2D_r == 1] = 2

    pre_2D = pre_2D_l + pre_2D_r
    # label_all = np.zeros(info_dict.image_shape_raw)
    label_all = np.zeros([1] + info_dict.image_shape_raw[1:])
    label_all[info_dict.index_floor:info_dict.index_ceil, :, :] = pre_2D

    # 反大图矫正
    label_all, info_dict = anti_check_large_image_zyx(label_all, info_dict)
    # 反体位矫正
    label_all = rotate_label_3D(label_all, info_dict.head_adjust_angle, info_dict.head_adjust_center,
                                   use_adjust=info_dict.use_head_adjust)

    # 根据训练的维度将维度复原
    # label_all = label_all.swapaxes(0, 2)
    label_all = np.array(label_all, np.uint8)
    print('Done to postprocess\n')

    return label_all, info_dict
