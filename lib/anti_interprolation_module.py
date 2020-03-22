#coding=utf-8

import numpy as np
import lib.assert_module as assertor
from lib.error import Error, ErrorCode
from lib.info_dict_module import InfoDict
from lib.interpolation_module import interp_3d_zyx, interp_2d_yx, CV2_interp_type
from lib.check_shape_module import check_shape,check_for_cv2_dtype
import cv2


def anti_interp_3d_zyx(input_array: np.ndarray, old_spacing, standard_spacing=(1, 1),
                       ori_shape=(512, 512), kind=CV2_interp_type.nearest, blur_kernal_size=0):
    """
    反插值图像回原来的图像大小
    整体流程:
    1. 判定输入是否为3维输入
    2. 对输入图像进行插值
    3. 如果ori_shape不为None则把反插值后的图像resize成ori_shape,保证输出图像和原始保持一致
    """
    # 1. 判定输入是否为3维输入
    assertor.array_x_dims_assert(input_array, 3, error_code=ErrorCode.pred_input_shape_error,
                                 msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(ori_shape, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(old_spacing, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(standard_spacing, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')

    if None in old_spacing or None in standard_spacing:
        print('     target_spacing is None and interpolation is discarded\n')
        return input_array

    if len(input_array.shape) != 3:
        print('The input volume should be 3 dimensional')
        raise ValueError

    if input_array.shape[1:] == ori_shape:
        return input_array

    #if input_array.shape[1] == ori_shape[0] and input_array.shape[2] == ori_shape[1]:
    # 2. 对输入图像进行插值
    row_size_new, col_size_new = ori_shape

    rst_array_list = [interp_mask_multi_labels(mask, col_size_new, row_size_new,
                                          blur_kernal_size) for mask in input_array]

    rst_array = np.stack(rst_array_list, axis=0)
    # rst_array = interp_3d_zyx(input_array, standard_spacing, old_spacing, kind=kind, kernel_size=blur_kernal_size)

    # 3. 如果ori_shape不为None则把反插值后的图像resize成ori_shape,保证输出图像和原始保持一致
    # if ori_shape is not None:
    #     rst_array = check_shape(rst_array, ori_shape)

    return rst_array


def anti_interp_4d_zyxc(input_array: np.ndarray, spacing, standard_spacing=(1, 1),
                         ori_shape=(512, 512), kind=CV2_interp_type.nearest, blur_kernal_size=0):
    """
    反插值图像回原来的图像大小
    整体流程:
    1. 判定输入是否为3维输入
    2. 对输入图像进行插值
    3. 如果ori_shape不为None则把反插值后的图像resize成ori_shape,保证输出图像和原始保持一致
    """
    # 1. 判定输入是否为3维输入
    assertor.array_x_dims_assert(input_array, 4, error_code=ErrorCode.pred_input_shape_error,
                                 msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(ori_shape, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(spacing, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')
    assertor.type_multi_assert(standard_spacing, [tuple, list], error_code=ErrorCode.pred_input_shape_error,
                               msg='Assert pos: anti_interprolation module')

    rst_list = []
    for i in range(input_array.shape[-1]):
        rst_array = anti_interp_3d_zyx(
            input_array[...,i], old_spacing=spacing, standard_spacing=standard_spacing,
            ori_shape=ori_shape, kind=kind, blur_kernal_size=blur_kernal_size)

        rst_list.append(np.expand_dims(rst_array, axis=3))

    rst_array = np.concatenate(rst_list, axis=3)

    return rst_array

def anti_interp_2d_pack(image_block, info_dict: InfoDict, kernal_size=0):
    '''2d image interpolation package
    :param image_block: 3d volume, format: zyx
    :param info_dict:  should have info_dict.image_before_interp
    :param kind: interpolation methods, cv2.INTER_LINEAR cv2.INTER_NEAREST cv2.INTER_CUBIC(slow)
    :param kernel_size: used in median blurring for interpolation results. if 0, then no blurring operation
    :return image_interp: resized image volume ,its dtype is same with image_block
    :return info_dict: info_dict
    '''

    if len(image_block.shape) != 3:
        # 输入图像的shape错误, 返回错误码
        Error.exit(ErrorCode.process_input_shape_error)

    if not "image_shape_before_interp" in info_dict:
        Error.exit(ErrorCode.process_module_error)

    raw_dtype = image_block.dtype
    image_block = check_for_cv2_dtype(image_block, raw_dtype)

    origin_x = info_dict.image_shape_before_interp[2]
    origin_y = info_dict.image_shape_before_interp[1]
    image_anti_interp = np.zeros((image_block.shape[0], origin_x, origin_y), np.float32)

    for i in range(image_block.shape[0]):
        image_one = interp_2d_yx(image_block[i, :, :], origin_x, origin_y, info_dict.interp_kind, kernal_size)
        image_anti_interp[i, :, :] = image_one

    return image_anti_interp, info_dict



def interp_mask_multi_labels(mask_2d, col_size_new, row_size_new, smooth_kernel = 0):
    """
    掩码插值，适应多值的情况。用cv2.INTER_LINEAR。
    如果用cv2.INTER_NEAREST会出现掩码整体往右下偏移的情况，应该尽量避免。
    :param mask_2d:
    :param col_size_new:
    :param row_size_new:
    :param smooth_kernel: 平滑核心的大小，None或者Int
    :return:
    """
    unique_value = np.unique(mask_2d)[1:]

    # from lib.utlis.visualization import plot2Image
    mask_return = np.zeros((row_size_new, col_size_new), dtype= np.uint8)
    # v = unique_value[0]
    for v in unique_value:
        mask_temp = np.array(mask_2d == v, dtype= np.uint8)
        mask_temp = cv2.resize(mask_temp, (col_size_new, row_size_new), interpolation=cv2.INTER_LINEAR)
        if smooth_kernel != 0:
            mask_temp = cv2.medianBlur(mask_temp, smooth_kernel)
        mask_temp = np.array(mask_temp>0.7, dtype= np.uint8) * v
        mask_return = np.amax(np.stack([mask_temp, mask_return], axis= 0), axis=0)
        # plot2Image(mask_return, mask_temp)
    # print(np.unique(mask_return))
    return mask_return

