#-*- coding:utf-8 -*-

import numpy as np
import cv2
from skimage.measure import label as label_function
from skimage.measure import regionprops
import math
from lib.error import Error, ErrorCode
from lib.rotate_module import rotate_3D
from lib.large_image_checker_module import check_large_image_zyx
from lib.interpolation_module import interp_3d_zyx, CV2_interp_type

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

def image_interp(data, target_size, interpolation):
    """插值函数（默认线性插值）
    # Arguments：
        data：待插值图像，三维数组
        target_size：插值后x、y的大小

    # Returns
        img_new：插值后的图像

    # Example
    """

    if len(np.shape(data)) != 3:
        print('DataError: the channel of data is not equal to 3')
        Error.exit(ErrorCode.process_input_shape_error)

    print('start interpolation......')

    z_old, rows_old, cols_old = np.shape(data)

    if len(target_size) == 2:
        rows_new = target_size[0]
        cols_new = target_size[1]
    elif len(target_size) == 1:
        rows_new = target_size[0]
        cols_new = target_size[0]
    else:
        rows_new = rows_old
        cols_new = cols_old

    img_new = np.zeros([z_old, rows_new, cols_new], dtype=np.float32)
    for i in range(z_old):
        # note: cv2.resize 函数的size输入为 宽(cols_new) * 高(rows_new)
        img_new[i, :, :] = cv2.resize(data[i, :, :], (cols_new, rows_new), interpolation=interpolation)

    print('complete interpolation......')
    return img_new

def judge_center(image, kind):
    """提取最大连通区域，在CT图像中就是将body提取出来
    # Arguments：
        image：图像
        kind：类型
    # Returns
        center：最大连通区域的中心坐标
        box或r：最大连通区域的box， 两个角坐标

    # Example
    """
    if kind == 'image':
        bb = image > -500
        cc = np.array(bb, dtype=np.uint8)
        d = label_function(cc, connectivity=2)

        e = regionprops(d)
        x, y = np.shape(d)
        length = x * 0.2
        f = []
        for i in range(len(e)):
            ee = e[i].centroid

            if ee[0] > length and ee[0] < x - length and ee[1] > length and ee[1] < x - length:

                f.append(e[i].area)
            else:
                f.append(0)

        ind = np.argmax(f)

        center = e[ind].centroid
        box = e[ind].bbox

        return center, box

    if kind == 'label':
        cc = np.array(image, dtype=np.uint8)
        d = label_function(cc, connectivity=2)

        e = regionprops(d)
        x, y = np.shape(d)
        length = x * 0.2
        f = []
        for i in range(len(e)):
            ee = e[i].centroid

            if ee[0] > length and ee[0] < x - length and ee[1] > length and ee[1] < x - length:

                f.append(e[i].area)
            else:
                f.append(0)

        ind = np.argmax(f)

        center = e[ind].centroid
        r = e[ind].equivalent_diameter

        return center, r

def Crop_3D(imgs, target_size, step, imgs_train):
    """
    对 img 和 label 按 x 轴为主轴按照 self.slice_size 和 self.step 切块

    :param imgs: 需要切块的 image
    :param labels: image 对应的 label
    :param target_size: 3D块Z大小
    :param step: 步进
    :param imgs_train: 堆叠的图像块
    :param labels_train: 堆叠的label块
    """
    print('----------     begin crop2.5D     --------')
    voxel_x = imgs.shape[0]
    target_gen_size = target_size

    # 计算切块需要循环的次数
    range_val = int(math.ceil((voxel_x - target_gen_size) / step) + 1)

    for i in range(range_val):
        start_num = i * step
        end_num = start_num + target_gen_size

        if end_num <= voxel_x:
            # 数据块长度没有超出x轴的范围,正常取块
            slice_img = imgs[np.newaxis, start_num:end_num, :, :]
        else:
            # 数据块长度超出x轴的范围, 从最后一层往前取一个 batch_gen_size 大小的块作为本次获取的数据块
            slice_img = imgs[np.newaxis, (voxel_x - target_gen_size):voxel_x, :, :]

        slice_img = np.transpose(slice_img, [0, 2, 3, 1])

        imgs_train = np.concatenate((imgs_train, slice_img), axis=0)
    print('----------     end crop2.5D     --------')
    return imgs_train

def im_train_crop(imgs, img_centre, crop_height, crop_width):
    # 判断原图的x与y是否大于裁剪后图像的大小
    z, height, width = np.shape(imgs)  # height 代表二维矩阵的行数  width 代表二维矩阵的列数
    judge = sum([height > crop_height, width > crop_width])
    imgs_new = []

    if judge == 2:  # 当原图大小大于裁剪后图像大小
        width_center = int(img_centre[1])
        height_center = int(img_centre[0])

        half_crop_width = int(crop_width / 2)

        half_crop_height_u = int(crop_height * 4 / 5)
        half_crop_height_d = int(crop_height * 1 / 5)

        width_l = width_center - half_crop_width
        width_r = width_center + half_crop_width

        height_u = height_center - half_crop_height_u
        height_b = height_center + half_crop_height_d

        if (width_l < 0):
            width_l = 0
            width_r = crop_width
        if (width_r > width):
            width_l = width - crop_width
            width_r = width
        if (height_u < 0):
            height_u = 0
            height_b = crop_height
        if (height_b > height):
            height_u = height - crop_height
            height_b = height

        four_corner = [height_u, height_b, width_l, width_r]
        for i, imgs_slice in enumerate(imgs):
            imgs_slice = imgs_slice[height_u:height_b, width_l:width_r]
            imgs_new.append(imgs_slice)
    elif judge == 0:  # 当原图大小小于裁剪后图像大小，则扩张
        image_std_new = np.ones([crop_height, crop_width], dtype=np.int32)  # 初始化为img的最小值，即背景
        if img_centre[0] - height / 2 < 0 or img_centre[0] + height / 2 > crop_height:
            if img_centre[0] - height / 2 < 0:
                height_u = 0
                height_b = int(height)
            else:
                height_u = int(crop_height - height)
                height_b = int(crop_height)
        else:
            height_u = int(img_centre[0] - height/ 2)
            height_b = int(img_centre[0] + height / 2)

        if img_centre[1] - width / 2 < 0 or img_centre[1] + width / 2 > crop_width:
            if img_centre[1] - width / 2 < 0:
                width_l = 0
                width_r = int(width)
            else:
                width_l = int(crop_width - width)
                width_r = int(crop_width)
        else:
            width_l = int(img_centre[1] - width / 2)
            width_r = int(img_centre[1] + width / 2)

        four_corner = [height_u, height_b, width_l, width_r]
        for i, imgs_slice in enumerate(imgs):
            image_std_new = np.min(imgs_slice) * image_std_new
            image_std_new[height_u:height_b, width_l:width_r] = imgs_slice
            imgs_new.append(image_std_new)
    else:
        Error.exit(ErrorCode.process_clips_out_of_range)
    # imgs_new与labels_new转换格式
    imgs_new = np.array(imgs_new, np.float32)

    return imgs_new, four_corner


def preprocess(image_raw, info_dict):

    print('\nBegin to preprocess')
    # 体位校正
    orig_data = rotate_3D(image_raw, info_dict.head_adjust_angle, info_dict.head_adjust_center,
                          use_adjust=info_dict.use_head_adjust)

    # 大图判定
    orig_data, info_dict = check_large_image_zyx(orig_data, info_dict)

    z_orig_data, rows_orig_data, cols_orig_data = np.shape(orig_data)
    info_dict.image_shape_raw = [z_orig_data, rows_orig_data, cols_orig_data]

    classes_value = np.array(info_dict.classes_value, np.int8)
    index = []
    classe_orgen = info_dict.organ_classify_pos
    for i, value in enumerate(classes_value):
        if value in classe_orgen:
            index.append(i)
    if len(index) == 0:
        print('there is no 3rd classification')
        Error.exit(ErrorCode.ld_no_target_layer)


    # 眼球、晶状体、视神经、视交叉、垂体，考虑到头部角度（仰头或低头）需向两侧各扩充3层
    if index[0] - 3 > 0:
        index_floor = index[0] - 3
    elif index[0] - 2 > 0:
        index_floor = index[0] - 2
    elif index[0] - 1 > 0:
        index_floor = index[0] - 1
    else:
        index_floor = index[0]

    if index[-1] + 2 < len(classes_value):
        index_ceil = index[-1] + 3
    elif index[-1] + 1 < len(classes_value):
        index_ceil = index[-1] + 2
    else:
        index_ceil = index[-1] + 1

    data2D = orig_data[index_floor:index_ceil + 2, :, :]  # 利用全身分类网络取出待分割器官所在层号的图像
    data2D[data2D > 2000] = 2000
    data2D[data2D < -1024] = -1024
    print('data.min(), data.max()', data2D.min(), data2D.max())
    data2D = (data2D - (-1024)) / (2000 - (-1024))
    data2D = np.array(data2D, np.float32)
    print('data.min(), data.max()', data2D.min(), data2D.max())
    data2D = (data2D - data2D.min()) / (data2D.max() - data2D.min())
    data2D = np.array(data2D, np.float32)

    info_dict.orig_images = data2D
    original_space = info_dict.spacing_list[1]
    if original_space <= 0.8 or original_space > 1.2:
        # rows_target_size = int(rows_orig_data * original_space / info_dict.target_spacing[0])     # 2D模型是插值到1.0
        # cols_target_size = int(cols_orig_data * original_space / info_dict.target_spacing[1])     # 2D模型是插值到1.0
        # data2D = image_interp(data2D, [rows_target_size, cols_target_size], cv2.INTER_LINEAR)
        data2D = interp_3d_zyx(data2D, info_dict.spacing_list[1:], info_dict.target_spacing, kind=CV2_interp_type.linear)

    size_2D = np.shape(data2D) # 插值后的size
    # STEP 3:裁剪
    img_judge = data2D[int((index_ceil - index_floor) / 2), :, :]
    img_centre, img_box = judge_center(img_judge, 'image')
    img_centre = list(np.array(img_centre, np.int16))
    # 裁剪成模型输入的大小 IMAGE_CROP_HEIGHT * IMAGE_CROP_WIDTH
    # data2D, four_corner = im_train_crop(data2D, img_centre, info_dict.model_input_size[0], info_dict.model_input_size[1])
    data2D = crop_by_size_and_shift(data2D, [320, 320], img_centre)

    info_dict.images = data2D

    # STEP 4: 标准化操作，减均值除方差

    # STEP 5: 制作网络的输入,shape为[z,x,y,1],2.5D
    imgs_train = np.zeros((1, 320, 320, 3))
    imgs_train = Crop_3D(data2D, 3, 1, imgs_train)
    imgs_train = imgs_train[1:, ...]
    # data2D = data2D.reshape(data2D.shape[0], data2D.shape[1], data2D.shape[2], 1)

    info_dict.size_2D = size_2D
    # info_dict.four_corner = four_corner
    info_dict.img_centre = img_centre
    info_dict.index_floor = index_floor
    info_dict.index_ceil = index_ceil

    return imgs_train, info_dict
