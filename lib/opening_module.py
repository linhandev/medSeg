#coding=utf-8

import cv2
import numpy as np
from skimage import morphology
from lib.error import ErrorCode
import lib.assert_module as assertor
from lib.shape_dict_module import MorpShapeCV2, MorpShapeSKI
from lib.cv2_dtype_checker_module import check_for_cv2_dtype, anti_check_for_cv2_dtype


def opening(input_array, opening_para, shape_type=MorpShapeCV2.ellipse):
    """
    开运算
    整体流程:
    1. 如果dtype在以下列表中则会报错，所以强转为float32
    2. 先得到element的形态，然后进行开运算
    3. 如果dtype在以下列表中则会报错，强转回raw_dtype
    注意:
    1. shape_type有几种选择：rect正方形,cross菱形,ellipse椭圆
    :param input_array: 输入的图像
    :param opening_para: opening核的大小
    :param shape_type: opening核的类型
    """
    # 断言保证
    assertor.type_assert(input_array, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: opening module')
    assertor.type_multi_assert(opening_para, [int, float],
                               error_code=ErrorCode.process_data_type_error, msg='Assert pos: opening module')
    assertor.type_assert(shape_type, int,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: opening module')
    assertor.array_bivalue_assert(input_array, msg='Assert pos: opening module')

    raw_dtype = input_array.dtype
    # 1. 如果dtype在以下列表中则会报错，所以强转为float32
    input_array = check_for_cv2_dtype(input_array, raw_dtype)

    # 2. 先得到element的形态，然后进行开运算
    element = cv2.getStructuringElement(shape_type, (opening_para, opening_para))
    rst_array = cv2.morphologyEx(input_array, cv2.MORPH_OPEN, element)

    # 3. 如果dtype在以下列表中则会报错，强转回raw_dtype
    rst_array = anti_check_for_cv2_dtype(rst_array, raw_dtype)

    return rst_array


def opening_ski(image, opening_para, shape_type=MorpShapeSKI.disk, is_binary=True):
    """
    开运算

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
    :param opening_para: 开运算系数
    :param is_binary: 输入的图像是否是二值化的, 如果是会调用 binary_opening 方法加速, 默认为 True
    :return: 处理后的图像
    """

    if is_binary:
        image = morphology.binary_opening(image, shape_type(opening_para))
    else:
        image = morphology.opening(image, shape_type(opening_para))

    return image


# if __name__ == '__main__':
#     data = np.zeros(shape=(256, 256), dtype=np.bool)
#     data[80:120, 80:120] = 1
#     from lkm_lib.utlis.visualization import plotOneImage
#     plotOneImage(data)
#     data[80:95,100:105] = 0
#     data[100:120,100:105] = 0
#     plotOneImage(data)
#     data[100:120, 80:100] = 2
#     new_data = opening(data, 10)
#     plotOneImage(new_data)
#     print(data.dtype)
