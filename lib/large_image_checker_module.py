#coding=utf-8

import numpy as np
import lib.assert_module as assertor
from lib.info_dict_module import InfoDict
from lib.error import Error, ErrorCode
from lib.interpolation_module import interp_2d_yx, CV2_interp_type


def trans_spacing_by_shape(old_spacing_2d, old_shape, new_shape):
    """
    根据图像的shape变化算出新的spacing数值
    计算公式：
    old_spacing / new_spacing = new_shape / old_shape
    new_spacing = (old_spacing * old_shape) / new_spacing
    整体流程：
    1. 先转换为numpy
    2. 计算新的spacing
    """
    # 断言保证
    assertor.type_multi_assert(old_spacing_2d, [np.ndarray, list, tuple], error_code=ErrorCode.process_data_type_error,
                               msg='Assert pos: trans_spacing_by_shape')
    assertor.type_multi_assert(old_shape, [np.ndarray, list, tuple], error_code=ErrorCode.process_data_type_error,
                               msg='Assert pos: trans_spacing_by_shape')
    assertor.type_multi_assert(new_shape, [np.ndarray, list, tuple], error_code=ErrorCode.process_data_type_error,
                               msg='Assert pos: trans_spacing_by_shape')

    # 1. 先转换为numpy
    if type(old_spacing_2d) in [tuple, list]:
        old_spacing_2d = np.array(old_spacing_2d)
    if type(old_shape) in [tuple, list]:
        old_shape = np.array(old_shape)
    if type(new_shape) in [tuple, list]:
        new_shape = np.array(new_shape)

    # 2. 计算新的spacing
    new_spacing_2d = list((old_spacing_2d * old_shape) / new_shape)

    return new_spacing_2d


def check_large_image_zyx(input_array: np.ndarray, info_dict: InfoDict, old_spacing=None, threshold_shape=(512, 512)):
    """
    功能:解决超过512, 512图像问题进行插值
    整体流程：
    1. 得到原始图像大小并进行判断
    2. 如果小于512, 直接返回
    3. 如果大于512, 计算两个维度的比值，选取大的比值计算出新的图像大小
    4. 计算出新的spacing, 存回info_dict
    5. 对原图像进行插值得到返回图像
    """
    if old_spacing is None:
        old_spacing = info_dict.spacing_list.copy()
    # 断言保证
    assertor.array_x_dims_assert(input_array, 3, error_code=ErrorCode.process_input_shape_error,
                                 msg='Assert pos: interp_large_image_zyx')
    assertor.equal_assert(len(old_spacing), 3, error_code=ErrorCode.process_input_shape_error,
                          msg='Assert pos: interp_large_image_zyx')
    assertor.equal_assert(len(threshold_shape), 2, error_code=ErrorCode.process_input_shape_error,
                          msg='Assert pos: interp_large_image_zyx')
    # 开关保护
    if not info_dict.use_large_image_check:
        return input_array, info_dict

    # 1. 得到原始图像大小并进行判断
    input_z, input_y, input_x = input_array.shape

    thre_y, thre_x = threshold_shape
    # 2. 如果小于512, 直接返回
    if (input_x <= thre_x) and (input_y <= thre_y):
        return input_array, info_dict
    # 3. 如果大于512, 计算两个维度的比值，选取大的比值计算出新的图像大小
    ratio_x = input_x / thre_x
    ratio_y = input_y / thre_y
    ratio = max(ratio_x, ratio_y)
    new_x = int(input_x / ratio)
    new_y = int(input_y / ratio)

    # 4. 计算出新的spacing, 存回info_dict
    new_spacing = trans_spacing_by_shape(old_spacing[1:], [input_y, input_x], [new_y, new_x])
    info_dict.spacing_list[1:] = new_spacing
    info_dict.large_raw_spacing = old_spacing

    # 5. 对原图像进行插值得到返回图像
    rst_array = np.zeros((input_z, new_y, new_x), dtype=input_array.dtype)
    for i in range(input_z):
        # 注意，这里需要反着调用
        rst_array[i,:,:] = interp_2d_yx(input_array[i,:,:], new_x, new_y)
    info_dict.large_image_shape = (input_z, input_y, input_x)
    info_dict.image_shape_raw = rst_array.shape

    return rst_array, info_dict


def anti_check_large_image_zyx(input_array: np.ndarray, info_dict: InfoDict):
    """
    功能: 反处理图像大于512,512的情况
    整体流程：
    1. 得到返回图像大小和原始图像大小并进行判断
    2. 如果为None, 直接返回
    3. 计算出新的spacing, 存回info_dict
    4. 对原图像进行插值得到返回图像
    """
    # 断言保证
    assertor.array_x_dims_assert(input_array, 3, error_code=ErrorCode.process_input_shape_error,
                                 msg='Assert pos: interp_large_image_zyx')

    # 开关保护
    if not info_dict.use_large_image_check:
        return input_array, info_dict

    # 1. 得到返回图像大小和原始图像大小并进行判断
    large_image_shape = info_dict.large_image_shape
    # 2. 如果为None, 直接返回
    if large_image_shape is None:
        return input_array, info_dict

    # 3. 计算出新的spacing, 存回info_dict
    large_raw_spacing = info_dict.large_raw_spacing
    if large_raw_spacing is None:
        print('large_image_shape exist, but large_raw_spacing is None')
        Error.exit(ErrorCode.process_data_type_error)
    info_dict.spacing_list = large_raw_spacing

    # 4. 对原图像进行插值得到返回图像
    rst_array = np.zeros(large_image_shape, dtype=input_array.dtype)
    input_z, input_y, input_x = large_image_shape
    for i in range(input_z):
        # 注意，这里需要反着调用
        rst_array[i, :, :] = interp_2d_yx(input_array[i, :, :], input_x, input_y, kind=CV2_interp_type.nearest)
    info_dict.image_shape_raw = large_image_shape

    return rst_array, info_dict


# if __name__ == '__main__':
#     info_dict = InfoDict()
#     info_dict.spacing_list = [1,1,1]
#     input_array = np.zeros((15,1024,512), dtype=np.int16)
#     out_array, info_dict = check_large_image_zyx(input_array, info_dict)
#     print(input_array.shape)
#     print(out_array.shape)
#     print(info_dict.spacing_list)
#     print(info_dict.large_image_shape)
#     print(info_dict.large_raw_spacing)
#     print()
#     new_array, info_dict = anti_check_large_image_zyx(out_array, info_dict)
#     print(input_array.shape)
#     print(out_array.shape)
#     print(new_array.shape)
#     print(info_dict.spacing_list)
#     print(info_dict.large_image_shape)
#     print(info_dict.large_raw_spacing)