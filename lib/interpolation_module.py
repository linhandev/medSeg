# coding:utf-8

import cv2
import time
import numpy as np
from lib.error import Error, ErrorCode
from lib.info_dict_module import InfoDict
import lib.assert_module as assertor
from lib.cv2_dtype_checker_module import check_for_cv2_dtype, anti_check_for_cv2_dtype

# from scipy import interpolate

# cv2 插值类型
class CV2_interp_type(object):
    linear = cv2.INTER_LINEAR
    nearest = cv2.INTER_NEAREST
    area = cv2.INTER_AREA
    cubic = cv2.INTER_CUBIC
    lanczos = cv2.INTER_LANCZOS4


def interp_3d_zyx(img_volume, spacing, new_spacing, kind=CV2_interp_type.linear, kernel_size=0,
                  is_inference=True):
    """
      interpolate on each slice of the image volume
img_volume
    :param img_volume: 3d volume, dimension order best be z rows cols, dtype best be np.int16
    :param spacing: (y,x) spacing
    :param new_spacing: (y,x) spacing
    :param kind: interpolation methods, cv2.INTER_LINEAR cv2.INTER_NEAREST cv2.INTER_CUBIC(slow)
    :param kernel_size: used in median blurring for interpolation results. if 0, then no blurring operation
    :return: resized image volume ,its dtype is same with img_volume
    """
    # 断言保证
    assertor.equal_assert(len(spacing), 2, error_code=ErrorCode.process_input_shape_error,
                          msg='Assert pos: interp_3d_zyx')
    assertor.equal_assert(len(new_spacing), 2, error_code=ErrorCode.process_input_shape_error,
                          msg='Assert pos: interp_3d_zyx')

    if is_inference:
        print('     Original volume shape is %s' % str(img_volume.shape))

    if None in spacing or None in new_spacing:
        print('     target_spacing is None and interpolation is discarded\n')
        return img_volume

    # 检查数据类型，并转换
    raw_dtype = img_volume.dtype
    img_volume = check_for_cv2_dtype(img_volume, raw_dtype)

    if len(img_volume.shape) != 3:
        # 输入图像的shape错误, 返回错误码
        Error.exit(ErrorCode.process_input_shape_error)

    img_volume = np.array(img_volume, dtype=img_volume.dtype)
    # z_size = img_volume.shape[0]
    # get new row,col from spacing  row - y, col - x
    row_size_new, col_size_new = trans_shape_by_spacing(img_volume.shape[1:], spacing, new_spacing)

    resize_image = np.zeros(shape=(img_volume.shape[0], row_size_new, col_size_new),dtype=img_volume.dtype)
    for i in range(img_volume.shape[0]):
        resize_image[i, :, :] = interp_2d_yx(img_volume[i,:,:],row_size_new,col_size_new,kind,kernel_size)
    # 使结果数据类型和输入一致

    resize_image = anti_check_for_cv2_dtype(resize_image, raw_dtype)
    if is_inference:
        print('     Shape after necessary interpolation is %s \n' % str(resize_image.shape))

    return resize_image


def interp_2d_yx (image_2d, row_size_new, col_size_new,kind=CV2_interp_type.linear, kernal_size=0):
    '''2d image interpolation
    :param image_2d: 2d volume, format: yx  
    :param row_size_new:   can be call "y"
    :param col_size_new:   can be call "x"
    :param kind: interpolation methods, cv2.INTER_LINEAR cv2.INTER_NEAREST cv2.INTER_CUBIC(slow)
    :param kernel_size: used in median blurring for interpolation results. if 0, then no blurring operation
    :return: resized image volume ,its dtype is same with image_2d
    '''
    if len(image_2d.shape) != 2:
        # 输入图像的shape错误, 返回错误码
        Error.exit(ErrorCode.process_input_shape_error)

    resize_slice = cv2.resize(image_2d, (col_size_new,row_size_new), interpolation=kind)
    resize_slice = resize_slice
    if kernal_size:
        # smoothes an image using the median filter
        image_new = cv2.medianBlur(resize_slice, kernal_size)
    else:
        image_new = resize_slice
    image_new = np.array(image_new, dtype=image_2d.dtype)

    return image_new


def trans_shape_by_spacing(old_shape_2d, spacing, new_spacing):
    '''
    based on shidejun'
    get new shape based on old spacing and new spacing
    :param old_shape_2d: 2-element list, nparray or tuple
    :param spacing: 2-element list, nparray or tuple
    :param new_spacing: 2-element list, nparray or tuple
    :return:
    '''
    resize_factor = np.array(spacing, dtype=np.float32) / np.array(new_spacing, dtype=np.float32)
    row_size = old_shape_2d[0]
    col_size = old_shape_2d[1]
    row_size_new = int(np.round(row_size * resize_factor[0]))
    col_size_new = int(np.round(col_size * resize_factor[1]))
    return row_size_new, col_size_new

def interp_2d_pack (image_block, info_dict: InfoDict, kernal_size = 0):
    '''2d image interpolation package
    :param image_block: 3d volume, format: zyx
    :param info_dict:  should have info_dict.spacing_list, info_dict.target_spacing
    :param kind: interpolation methods, cv2.INTER_LINEAR cv2.INTER_NEAREST cv2.INTER_CUBIC(slow)
    :param kernel_size: used in median blurring for interpolation results. if 0, then no blurring operation
    :return image_interp: resized image volume ,its dtype is same with image_block
    :return info_dict: info_dict
    '''

    if len(image_block.shape) != 3:
        # 输入图像的shape错误, 返回错误码
        Error.exit(ErrorCode.process_input_shape_error)

    if not "target_spacing" in info_dict:
        Error.exit(ErrorCode.process_module_error)

    raw_dtype = image_block.dtype
    image_block = check_for_cv2_dtype(image_block, raw_dtype)

    spacing = [info_dict.spacing_list[1], info_dict.spacing_list[2]]
    pixel = np.array(spacing, dtype=np.float32) / np.array(info_dict.target_spacing, dtype=np.float32)
    new_x = int(image_block.shape[2] * pixel[0])
    new_y = int(image_block.shape[1] * pixel[1])

    info_dict.image_shape_before_interp = image_block.shape

    image_interp = np.zeros((image_block.shape[0], new_x, new_y), np.float32)

    for i in range(image_block.shape[0]):
        image_one = interp_2d_yx(image_block[i, :, :], new_x, new_y, info_dict.interp_kind, kernal_size)
        image_interp[i, :, :] = image_one

    return image_interp,info_dict


# if __name__ == '__main__':
#
#     import os
#     import matplotlib.pyplot as plt
#     from lkm_lib.utlis.visualization import plotOneImage
#
#     root = r'G:\\000_test_img'
#     file = r'image_raw.npy'  # image_raw - > test3d
#     file_path = os.path.join(root, file)
#     image_3d = np.load(file_path)
#
#     image_2d = image_3d[5,:,:]
#     # print(image_3d.shape)
#     print (image_2d.shape)
#     # image_3d = np.array(np.round(image_3d), dtype=np.uint8) # int64 uint16
#     image_3d = np.array(np.round(image_3d), dtype=np.uint8)
#     # print('before resize:',image_3d.dtype)
#     print('before resize dtype:',image_2d.dtype)
#     #test_resize = interp_2d_yx (image_2d, 120, 80,CV2_interp_type.linear,5)
#     test_resize = interp_3d_zyx(image_3d, [1.0,1.0], [0.5,0.25], CV2_interp_type.linear)
#
#     print('after resize:', test_resize.dtype)
#     print('after resize shape:', test_resize.shape)
#
#     plotOneImage(test_resize[5,:,:])
#     # plt.imshow(test_resize)
#     # plt.show()


