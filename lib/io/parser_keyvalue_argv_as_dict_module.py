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


def _parser_str_as_dict(input_str: str):
    key, value = input_str.split('=')

    rst_dict = {}
    value = str(value).strip()
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if (value=='True') or (value=='true'):
                value = True
            elif (value=='False') or (value=='false'):
                value = False
            else:
                value = str(value)

    rst_dict[key] = value

    return rst_dict


def parser_keyvalue_argv_as_dict(input_argv: list, file_pos: str):
    """
    转换key,value输入为字典
    整体流程：
    1. 遍历每一个输入信息
    2. 没有key=value形式的输入则跳过
    3. 解析当前key=value输入为暂存的字典
    4. 放入到整体字典中
    :param input_argv:
    :return:
    """
    # 1. 遍历每一个输入信息
    rst_dict = {}
    for input_str in input_argv:
        # 2. 没有key=value形式的输入则跳过
        if '=' not in input_str:
            continue
        # 3. 解析当前key=value输入为暂存的字典
        temp_dict = _parser_str_as_dict(input_str)
        # 4. 放入到整体字典中
        for key, value in temp_dict.items():
            rst_dict[key] = value

    if 'model_path' not in rst_dict.keys():
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(file_pos)))),
                                      'model')
        rst_dict['model_path'] = model_path
    return rst_dict
