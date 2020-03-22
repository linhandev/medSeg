# coding:utf-8

import numpy as np
import cv2
from lib.error import Error, ErrorCode
import lib.assert_module as assertor
from lib.cv2_dtype_checker_module import check_for_cv2_dtype, anti_check_for_cv2_dtype

def check_shape(image, standar_shape=(512, 512),interp_kind =cv2.INTER_NEAREST):
    """
    检查图像尺寸，如果不为设定大小(512,512),则调整到设定尺寸
    :param1： image - 待检测尺寸,shape = 3
    param2: standar_shape 标准尺寸
    :param3: cv2插值方式
    :return: 调整后的图像大小
    """
    assertor.type_assert(image, np.ndarray,
                         error_code=ErrorCode.process_data_type_error, msg='Assert pos: check_shape module')
    img_shape = image.shape
    raw_dtype = image.dtype
    image = check_for_cv2_dtype(image, raw_dtype)

    if len(img_shape) != 3:
        # 输入图像的shape错误, 返回错误码
        Error.exit(ErrorCode.process_input_shape_error)

    if img_shape[1] == standar_shape[0] and img_shape[2] == standar_shape[1]:
        return image

    resize_image = np.zeros(shape=(image.shape[0], standar_shape[0], standar_shape[1]),dtype=image.dtype)
    for i in  range (image.shape[0]):
        resize_image[i,:,:] = cv2.resize(image[i,:,:], (standar_shape[1],standar_shape[0]), interpolation=interp_kind)

    resize_image = anti_check_for_cv2_dtype(resize_image, raw_dtype)

    return resize_image


# if __name__ == '__main__':
#
#     test_data = np.zeros(shape=(3,1024, 1024),dtype=np.uint32)  #np.uint16
#     image = check_shape(test_data,(512,512))
#     print('dtype:',image.dtype)
#     print(image.shape)