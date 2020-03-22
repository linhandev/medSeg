#coding=utf-8

import cv2
import numpy as np
from skimage import morphology
from lib.error import ErrorCode
import lib.assert_module as assertor
from lib.shape_dict_module import MorpShapeCV2, MorpShapeSKI
from lib.cv2_dtype_checker_module import check_for_cv2_dtype, anti_check_for_cv2_dtype


def closing(input_array, closing_para, shape_type=MorpShapeCV2.ellipse):
    """
    闭运算
    整体流程:
    1. 如果dtype在以下列表中则会报错，所以强转为float32
    2. 先得到element的形态，然后进行闭运算
    3. 如果dtype在以下列表中则会报错，强转回raw_dtype
    注意:
    1. shape_type：rect正方形,cross菱形,ellipse椭圆
    :param input_array: 输入的图像
    :param closing_para: closing核的大小
    :param shape_type: closing核的类型
    """
    # 断言保证
    assertor.type_assert(input_array, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: closing module')
    assertor.type_multi_assert(closing_para, [int, float],
                               error_code=ErrorCode.process_data_type_error, msg='Assert pos: closing module')
    assertor.type_assert(shape_type, int,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: closing module')
    assertor.array_bivalue_assert(input_array, msg='Assert pos: closing module')

    raw_dtype = input_array.dtype
    # 1. 如果dtype在以下列表中则会报错，所以强转为float32
    input_array = check_for_cv2_dtype(input_array, raw_dtype)

    # 2. 先得到element的形态，然后进行闭运算
    element = cv2.getStructuringElement(shape_type, (closing_para, closing_para))
    rst_array = cv2.morphologyEx(input_array, cv2.MORPH_CLOSE, element)

    # 3. 如果dtype在以下列表中则会报错，强转回raw_dtype
    rst_array = anti_check_for_cv2_dtype(rst_array, raw_dtype)

    return rst_array


def closing_ski(image, closing_para, shape_type=MorpShapeSKI.disk, is_binary=True):
    """
    闭运算

    :param image: 输入的图像
    :param shape_type: { 'square'  # 正方形
                    'disk' # 平面圆形
                    'ball'  # 球形
                    'cube' # 立方体形
                    'diamond' # 钻石形
                    'rectangle' # 矩形
                    'star'   # 星形
                    'octagon' # 八角形
                    'octahedron' # 八面体
                    }
    :param closing_para: 闭运算系数
    :param is_binary: 输入的图像是否是二值化的, 如果是会调用 binary_closing 方法加速, 默认为 True
    :return: 处理后的图像
    """

    if is_binary:
        image = morphology.binary_closing(image, shape_type(closing_para))
    else:
        image = morphology.closing(image, shape_type(closing_para))

    return image


# if __name__ == '__main__':
#     data = np.zeros(shape=(256, 256), dtype=np.int)
#     data[80:120, 80:120] = 1
#     import yy_lib as yy
#     yy.shower.show_np(data)
#     data[80:95,100:105] = 0
#     data[100:120,100:105] = 0
#     yy.shower.show_np(data)
#     # data[100:120, 80:100] = 2
#     new_data = closing(data, 10)
#     yy.shower.show_np(new_data)
#     print(data.dtype)
