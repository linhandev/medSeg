#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: parser_argv_as_dict_module.py
@Time: 2018-12-04 14:40
@Last_update: 2018-12-04 14:40
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os


def _parser_argv_as_dict(input_argv: list, is_classify=False):
    rst_dict = {}
    if is_classify:
        rst_dict['data_path'] = input_argv[0]
        rst_dict['goal_path'] = input_argv[1]
        rst_dict['model_path'] = input_argv[2]
    else:
        rst_dict['data_path'] = input_argv[0]
        rst_dict['classes_path'] = input_argv[1]
        rst_dict['goal_path'] = input_argv[2]
        rst_dict['model_path'] = input_argv[3]

    return rst_dict


def parser_classify_argv_as_dict(input_argv: list):
    return _parser_argv_as_dict(input_argv, is_classify=True)


def parser_argv_as_dict(argv_list: list, file_pos=None, is_classify=False):
    """
    老版linkone解析方式
    判别输入个数，再根据是否是分类算法进行路径解析，存储到info_dict中
    :param argv_list:
    :param is_classify:
    :return:
    """
    # 如果传参为4个，则使用原始的分割调用方法
    if len(argv_list) == 4:
        input_dict = _parser_argv_as_dict(argv_list, is_classify)
    # 如果传参是3个
    elif len(argv_list) == 3:
        # 是调用分类网络，则使用的是原始分类调用方法
        if is_classify:
            input_dict = _parser_argv_as_dict(argv_list, is_classify)
        # 是分割网络，则使用的是成都新的分割3接口方法
        else:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(file_pos)))),
                                      'model')
            input_dict = _parser_argv_as_dict(argv_list + [model_path], is_classify)
    # 如果传参是2个，则是成都新的分类2接口方法
    elif len(argv_list) == 2:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(file_pos)))),
                                  'model')
        input_dict = _parser_argv_as_dict(argv_list + [model_path], is_classify)
    # 其他则为参数错误
    else:
        raise ValueError('parameter number is wrong...')

    return input_dict