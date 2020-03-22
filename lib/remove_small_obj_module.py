#coding=utf-8

import numpy as np
from skimage import morphology
from lib.error import ErrorCode
import lib.assert_module as assertor


def remove_small_obj(input_array: np.ndarray, min_size=64, connectivity=1):
    """
    删除小块区域, 默认调用 skimage 的方法
    整体流程:
    1. 判别输入矩阵是否为bool矩阵，如果不是则记录dtype到raw_dtype并转换为bool矩阵
    2. 调用去除小物体的函数
    3. 如果输入矩阵不是bool矩阵则转换为raw_dtype返回
    :param input_array: 数组矩阵
    :param min_size: 最小连通区域尺寸，小于该尺寸的都将被删除. 默认为64. int型
    :param connectivity: 邻接模式，1表示4邻接，2表示8邻接. 默认为1. int型
    :return: 处理后的图像数组
    """
    # 断言保证
    assertor.type_assert(input_array, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_obj module')
    assertor.type_multi_assert(min_size, [int, np.int8, np.int16, np.int32, np.int64, float, np.float32, np.float64],
                               error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_obj module')
    assertor.type_assert(connectivity, int,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: remove_small_obj module')

    # 1. 判别输入矩阵是否为bool矩阵，如果不是则记录dtype到raw_dtype并转换为bool矩阵
    raw_dtype = input_array.dtype
    if raw_dtype not in [bool, np.bool]:
        input_array = input_array.astype(np.bool)

    # 2. 调用去除小物体的函数
    rst_array = morphology.remove_small_objects(ar=input_array, min_size=min_size, connectivity=connectivity)

    # 3. 如果输入矩阵不是bool矩阵则转换为raw_dtype返回
    if raw_dtype not in [bool, np.bool]:
        rst_array = rst_array.astype(raw_dtype)

    return rst_array


# if __name__ == '__main__':
#     data = np.zeros(shape=(256, 256), dtype=np.uint8)
#     data[80:120, 80:120] = 1
#     import yy_lib as yy
#     yy.shower.show_np(data)
#     data[60:65,70:75] = 1
#     data[150:155,160:165] = 1
#     yy.shower.show_np(data)
#     # data[100:120, 80:100] = 2
#     new_data = remove_small_obj(data, 100)
#     yy.shower.show_2np(data, new_data)
#     print(data.dtype)
