#coding=utf-8

import numpy as np
from skimage import morphology
from lib.error import ErrorCode
import lib.assert_module as assertor


def remove_small_holes(input_array: np.ndarray, area_threshold=64, connectivity=1):
    """
    移除小于指定尺寸的连续孔
    整体流程:
    1. 判别输入矩阵是否为bool矩阵，如果不是则记录dtype到raw_dtype并转换为bool矩阵
    2. 调用去除小洞的函数
    3. 如果输入矩阵不是bool矩阵则转换为raw_dtype返回
    :param input_array: 待操作的数组
    :param area_threshold: 以像素为单位的连续孔将被填充的最大面积.默认64, int类型
    :param connectivity: 邻接模式，1表示4邻接，2表示8邻接. 默认为1, int类型
    :param is_change_input: 如果不改变原矩阵，则会重新拷贝一遍，默认为直接更改, 如果要不改变原数据则设is_change_input为False
    :return: 处理后的图像数组
    """
    # 断言保证
    assertor.type_assert(input_array, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_holes module')
    assertor.type_multi_assert(area_threshold, [int, np.int8, np.int16, np.int32, np.int64, float, np.float32],
                               error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_holes module')
    assertor.type_assert(connectivity, int,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_holes module')

    # 1. 判别输入矩阵是否为bool矩阵，如果不是则记录dtype到raw_dtype并转换为bool矩阵
    raw_dtype = input_array.dtype
    if raw_dtype not in [bool, np.bool]:
        input_array = input_array.astype(np.bool)

    # 2. 调用去除小洞的函数
    rst_array = morphology.remove_small_holes(ar=input_array, min_size=area_threshold, connectivity=connectivity)

    # 3. 如果输入矩阵不是bool矩阵则转换为raw_dtype返回
    if raw_dtype not in [bool, np.bool]:
        rst_array = rst_array.astype(raw_dtype)

    return rst_array


# if __name__ == '__main__':
#     data = np.zeros(shape=(256, 256), dtype=np.float32)
#     data[80:120, 80:120] = 1
#     import yy_lib as yy
#     yy.shower.show_np(data)
#     data[90:95,100:105] = 0
#     data[100:110,100:105] = 0
#     yy.shower.show_np(data)
#     # data[100:120, 80:100] = 2
#     new_data = remove_small_holes(data, 100)
#     yy.shower.show_2np(data, new_data)
#     print(data.dtype)
