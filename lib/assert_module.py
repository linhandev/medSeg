#coding=utf-8

import os
import inspect
import numpy as np
from lib.error import Error


# TODO 风险: 如果b=a,然后传入b为传参,会显示最开始的变量名也就是a
def get_varname(var: object):
    """
    获取变量名
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def type_assert(input_: object, type_: type, error_code=None, msg=None):
    """
    断言输入的类型
    """
    if type(input_) != type_:
        print('The input(name:%s) type not meet the required, \nexpect type:%s, input type:%s' % (
            get_varname(input_), str(type_), str(type(input_))))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def type_multi_assert(input_: object, type_list: list, error_code=None, msg=None):
    """
    断言输入类型为type list内的类型
    """
    # 断言保证
    list_type_assert(type_list, type, error_code=error_code)
    input_type = type(input_)
    if input_type not in type_list:
        print('The input(name:%s) not meet the required, \nexpect type:%s, input type:%s' % (
            get_varname(input_), str(type_list), str(input_type)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def instance_of_assert(input_: object, type_: type, error_code=None, msg=None):
    """
    断言输入的类型是子类
    """
    if not isinstance(input_, type_):
        print('The input(name:%s) not meet the required, \nexpect type:%s has instance type input type:%s' % (
            get_varname(input_), str(type_), str(type(input_))))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def method_not_override_assert():
    """
    没有重写的错误
    """
    if True:
        print('The method meet problem, this method should be override!')


def condition_not_happen_assert(info='', error_code=None, msg=None):
    """
    条件不应该存在的断言
    """
    print(info)
    if True:
        print('The condition meet problem, this condition should not happen!')
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def equal_assert(input_1: object, input_2: object, error_code=None, msg=None):
    """
    保证input_1 等于 input_2
    """
    if input_1 != input_2:
        print('The input1(name:%s), input2(name:%s) not meet the required, \n'
              'expect input1 equal to input2, \ninput1:%s, input2:%s' % (
               get_varname(input_1), get_varname(input_2),
               str(input_1), str(input_2)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def equal_multi_assert(input_1: object, input_2_list: list, error_code=None, msg=None):
    """
    保证input_1 等于 input_2_list中的值
    """
    type_assert(input_2_list, list, error_code=error_code)
    rst_bool = False
    for input_2 in input_2_list:
        temp_bool = (input_1 == input_2)
        rst_bool = rst_bool or temp_bool
    if not rst_bool:
        print('The input1(name:%s), input2(name:%s) not meet the required, \n'
              'expect input1 equal to the value in input2 list, \ninput1:%s, input2:%s' % (
               get_varname(input_1), get_varname(input_2),
               str(input_1), str(input_2)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def not_equal_assert(input_1: object, input_2: object, error_code=None, msg=None):
    """
    保证input_1 不等于 input_2
    """
    if input_1 == input_2:
        print('The input1(name:%s), input2(name:%s) not meet the required, \n'
              'expect input1 not equal to input2, \ninput1:%s, input2:%s' % (
               get_varname(input_1), get_varname(input_2),
               str(input_1), str(input_2)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def not_None_assert(input_1: object, error_code=None, msg=None):
    """
    保证input_1 不等于 None
    """
    if input_1 is None:
        print('The input(name:%s) not meet the required, \n'
              'expect input1 is not None, \ninput1:%s' % (
               get_varname(input_1), str(input_1)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def greater_or_equal_assert(input_1: object, input_2: object, error_code=None, msg=None):
    """
    保证input_1 大于等于input_2
    """
    if input_1 < input_2:
        print('The input1(name:%s), input2(name:%s) not meet the required, \n'
              'expect input1 greater or equal than input2, \ninput1:%s, input2:%s' % (
               get_varname(input_1), get_varname(input_2),
               str(input_1), str(input_2)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def smaller_or_equal_assert(input_1: object, input_2: object, error_code=None, msg=None):
    """
    保证input_1 小于等于input_2
    """
    if input_1 > input_2:
        print('The input1(name:%s), input2(name:%s) not meet the required, \n'
              'expect input1 smaller or equal than input2, \ninput1:%s, input2:%s' % (
               get_varname(input_1), get_varname(input_2),
               str(input_1), str(input_2)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def array_x_dims_assert(input_array: np.ndarray, dims: int, error_code=None, msg=None):
    """
    保证输入矩阵为x维矩阵
    """
    # 断言输入保证
    type_assert(input_array, np.ndarray, error_code=error_code)
    type_assert(dims, int, error_code=error_code)

    array_dim = len(input_array.shape)
    if array_dim != dims:
        print('The input(name:%s) not meet the required, \nexpect dims:%s, input dims:%s' % (
            get_varname(input_array), str(dims), str(array_dim)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


# 待测试
def array_x_dims_multi_assert(input_array: np.ndarray, dims_list: list, error_code=None, msg=None):
    """
    保证输入矩阵为x维矩阵,x为在dims_list中的数据
    """
    # 断言输入保证
    type_assert(input_array, np.ndarray, error_code=error_code)
    type_assert(dims_list, list, error_code=error_code)

    array_dim = len(input_array.shape)
    if array_dim not in dims_list:
        print('The input(name:%s) not meet the required, \nexpect dims:%s, input dims:%s' % (
            get_varname(input_array), str(dims_list), str(array_dim)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def byxz_seq_array_assert(input_: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照byxz顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # # 断言输入保证
    # type_assert(input_, np.ndarray, error_code=error_code)
    # array_x_dims_assert(input_, 4, error_code=error_code)
    #
    # axis_y, axis_x = input_.shape[1:3]
    # if axis_y != axis_x:
    #     'The input(name:%s) not meet the required, \nexpect axis seq is byxz and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_), str(input_.shape))
    print('This assert will be discarded')


def zyx_seq_array_assert(input_: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照zyx顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # # 断言输入保证
    # type_assert(input_, np.ndarray)
    # array_x_dims_assert(input_, 3)
    #
    # axis_y, axis_x = input_.shape[1:]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is zyx and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_), str(input_.shape))
    print('This assert will be discarded')


def zyx_seq_image_assert(input_image: np.ndarray, error_code=None, msg=None):
    """
    保证输入图像为按照zyx顺序排列,
    当前的断言方法是判断y,x轴大小相同
    同时dtype为float32
    """
    # 断言输入保证
    # array_dtype_assert(input_image, np.float32)
    # zyx_seq_array_assert(input_image)
    print('This assert will be discarded')


def zyx_seq_mask_assert(input_mask: np.ndarray, error_code=None, msg=None):
    """
    保证输入图像为按照zyx顺序排列,
    当前的断言方法是判断y,x轴大小相同
    同时dtype为int8
    """
    # 断言输入保证
    # array_dtype_assert(input_mask, np.int8)
    # zyx_seq_array_assert(input_mask)
    print('This assert will be discarded')


def xyz_seq_array_assert(input_: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照xyz顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # 断言输入保证
    # type_assert(input_, np.ndarray)
    # array_x_dims_assert(input_, 3)
    #
    # axis_y, axis_x = input_.shape[:2]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is xyz and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_), str(input_.shape))
    print('This assert will be discarded')


# 待测试
def xyc_seq_array_assert(input_: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照xyc顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # 断言输入保证
    # type_assert(input_, np.ndarray)
    # array_x_dims_assert(input_, 3)
    #
    # axis_y, axis_x = input_.shape[:2]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is xyc and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_), str(input_.shape))
    print('This assert will be discarded')


# 待测试
def xyzc_seq_array_assert(input_: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照xyzc顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # 断言输入保证
    # type_assert(input_, np.ndarray)
    # array_x_dims_assert(input_, 4)
    #
    # axis_y, axis_x = input_.shape[:2]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is xyzc and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_), str(input_.shape))
    print('This assert will be discarded')


def yx_seq_array_assert(input_array: np.ndarray, error_code=None, msg=None):
    """
    保证输入矩阵为按照yx顺序排列,
    当前的断言方法是判断y,x轴大小相同
    """
    # 断言输入保证
    # type_assert(input_array, np.ndarray)
    # array_x_dims_assert(input_array, 2)
    #
    # axis_y, axis_x = input_array.shape[:]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is yx and y=x,' \
    #     ' input shape:%s' \
    #     % (get_varname(input_array), str(input_array.shape))
    print('This assert will be discarded')


def list_length_assert(input_: list, length: int, error_code=None, msg=None):
    """
    保证输入矩阵为x维矩阵
    """
    # 断言输入保证
    type_assert(input_, list, error_code=error_code)

    list_length = len(input_)
    if list_length != length:
        print('The input(name:%s) not meet the required, \n'
              'expect list length:%s, input list length:%s' % (
               get_varname(input_), str(length), str(list_length)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def list_type_assert(input_list: list, type_: type, error_code=None, msg=None):
    """
    保证输入list为type的list类型
    """
    # 断言保证
    type_assert(input_list, list, error_code=error_code)

    for item in input_list:
        type_assert(item, type_, error_code=error_code)


def in_list_assert(item: object, input_list: list, error_code=None, msg=None):
    """
    保证item在input list中
    """
    # 断言保证
    type_assert(input_list, list, error_code=error_code)

    if item not in input_list:
        print('The %s,%s not meet the required, \n'
              'expect item in input list,\nitem :%s, input_list: %s' % (
               get_varname(item), get_varname(input_list),
               str(item), str(input_list)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def dict_has_key_assert(key: object, input_dict: dict, error_code=None, msg=None):
    """
    保证key在input dict的key中
    """
    # 断言保证
    type_assert(input_dict, dict, error_code=error_code)

    if key not in list(input_dict.keys()):
        print('The %s,%s not meet the required, \n'
              'expect key in input dict,\nkey :%s, input_dict: %s' % (
               get_varname(key), get_varname(input_dict),
               str(key), str(input_dict)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def array_length_assert(input_: np.ndarray, length: int, axis=0, error_code=None, msg=None):
    """
    保证输入矩阵的axis维长度为length
    """
    # 断言输入保证
    type_assert(input_, np.ndarray, error_code=error_code)
    greater_or_equal_assert(len(input_.shape), axis, error_code=error_code)

    list_length = input_.shape[axis]
    if list_length != length:
        print('The input(name:%s) not meet the required, \n'
              'expect array length:%s, input array length:%s' % (
               get_varname(input_), str(length), str(list_length)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def array_shape_assert(input_array: np.ndarray, shape: tuple, error_code=None, msg=None):
    """
    保证输入矩阵的shape
    """
    # 断言保证
    type_assert(input_array, np.ndarray, error_code=error_code)
    tuple_type_assert(shape, int, error_code=error_code)
    tuple_length_assert(shape, len(input_array.shape), error_code=error_code)

    array_shape = input_array.shape
    if array_shape != shape:
        print('The input(name:%s) not meet the required, \n'
              'expect array shape:%s, input array shape:%s' % (
               get_varname(input_array), str(shape), str(array_shape)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def array_dtype_assert(input_array: np.ndarray, dtype_: type, error_code=None, msg=None):
    """
    保证输入矩阵的dtype
    """
    # 断言保证
    type_assert(input_array, np.ndarray, error_code=error_code)
    type_assert(dtype_, type, error_code=error_code)

    array_dtype = input_array.dtype
    if array_dtype != dtype_:
        print('The input(name:%s) not meet the required, \n'
              'expect array dtype:%s, input array dtype:%s' % (
               get_varname(input_array), str(dtype_), str(array_dtype)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def zyx_seq_space_assert(input_space: np.ndarray, error_code=None, msg=None):
    """
    保证输入space为按照zyx顺序排列,
    当前的断言方法是判断y,x轴大小相同
    并且dtype是np.float32
    """
    # 断言输入保证
    # array_length_assert(input_space, 3)
    # array_dtype_assert(input_space, np.float32)
    #
    # axis_y, axis_x = input_space[1:]
    # assert axis_y == axis_x, \
    #     'The input(name:%s) not meet the required, \nexpect axis seq is zyx and y=x,' \
    #     ' input :%s' \
    #     % (get_varname(input_space), str(input_space))
    print('This assert will be discarded')


def path_exist_assert(input_path: str, error_code=None, msg=None):
    """
    保证输入路径存在
    """
    is_exist = os.path.exists(input_path)
    if not is_exist:
        print('The input(name:%s) is not exist, path: %s' % (
               get_varname(input_path), str(input_path)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def file_exist_assert(input_file: str, error_code=None, msg=None):
    """
    保证输入路径存在
    """
    is_exist = os.path.exists(input_file)
    if not is_exist:
        print('The input(name:%s) is not exist, file: %s' % (
               get_varname(input_file), str(input_file)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def file_not_exist_assert(input_file: str, error_code=None, msg=None):
    """
    保证输入路径不存在
    """
    is_exist = os.path.exists(input_file)
    if is_exist:
        print('The input(name:%s) is exist, file: %s' % (
            get_varname(input_file), str(input_file)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def tuple_type_assert(input_tuple: tuple, type_: type, error_code=None, msg=None):
    """
    保证输入tuple为type的tuple类型
    """
    # 断言保证
    type_assert(input_tuple, tuple, error_code=error_code)

    for item in input_tuple:
        type_assert(item, type_, error_code=error_code)


def tuple_length_assert(input_tuple: tuple, length: int, error_code=None, msg=None):
    """
    保证输入tuple的长度为length
    """
    # 断言保证
    type_assert(input_tuple, tuple, error_code=error_code)
    type_assert(length, int, error_code=error_code)

    tuple_length = len(input_tuple)
    if tuple_length != length:
        print('The input(name:%s) not meet the required, \n'
              'expect tuple length:%s, input tuple length:%s' % (
               get_varname(input_tuple), str(length), str(tuple_length)))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)


def array_bivalue_assert(input_array: np.ndarray, error_code=None, msg=None):
    """
    保证输入tuple的长度为length
    """
    # 断言保证
    type_assert(input_array, np.ndarray, error_code=error_code)

    value_num = len(np.unique(input_array))
    if value_num > 2:
        print('The input array(name:%s) is not bi value array, \n'
              'input array values:%s' % (
               get_varname(input_array), str(np.unique(input_array))))
        if error_code is not None:
            Error.warn(error_code)
        if msg is not None:
            print(msg)
