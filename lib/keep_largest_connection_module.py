#coding=utf-8

import numpy as np
from skimage.measure import regionprops, label
from lib.error import ErrorCode
import lib.assert_module as assertor
from lib.remove_small_obj_module import remove_small_obj


def keep_largest_connection(input_array: np.ndarray, max_percent=0.99):
    """
    保留最大连通区域，同时支持2d，3d的图像
    整体流程：
    1. 判定输入矩阵最大值是否为零，如果为零则表示为空矩阵，无最大连通区域
    2. 首先找到所有联通区域
    3. 找到所有联通区域的最大值
    4. 去除小于最大值的区域
    注意：max_percent设定为去除最大区域的大小的百分比, 默认为0.99因为如果为1.0有时会有问题
    """
    # 断言保证
    assertor.type_assert(input_array, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: keep_largest_connection module')
    assertor.type_assert(max_percent, float, error_code=ErrorCode.process_data_type_error,
                         msg='Assert pos: keep_largest_connection module')

    # 1. 判定输入矩阵最大值是否为零，如果为零则表示为空矩阵，无最大连通区域
    if np.max(input_array) == 0:
        print('Keep_largest_connection, array is empty, no max area')
        rst_array = input_array
    else:
        # 2. 首先找到所有联通区域
        label_data = label(input_array)
        return_property = regionprops(label_data)
        area_list = [i.area for i in return_property]
        # 3. 找到所有联通区域的最大值
        area_max = np.max(area_list)
        # 4. 去除小于最大值的区域
        rst_array = remove_small_obj(label_data, min_size=area_max*max_percent, connectivity=1)

        rst_array = rst_array.astype(input_array.dtype)

    return rst_array


# if __name__ == '__main__':
#     data = np.zeros(shape=(256, 256), dtype=np.float32)
#     data[80:120, 80:120] = 1
#     import yy_lib as yy
#     data[60:85,70:75] = 1
#     data[150:155,160:165] = 1
#     data[80:85, 75:80] = 1
#     data[110:115, 200:220] = 1
#     yy.shower.show_np(data)
#     # data[100:120, 80:100] = 2
#     print(data.dtype)
#     new_data = keep_largest_connection(data)
#     yy.shower.show_2np(data, new_data)
#     print(data.dtype)
