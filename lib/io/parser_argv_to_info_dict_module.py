#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: parser_argv_to_info_dict.py
@Time: 2019-01-21 15:40
@Last_update: 2019-01-21 15:40
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import sys
import lib.assert_module as assertor
from lib.info_dict_module import InfoDict
from lib.error import Error, ErrorCode
from lib.io import parser_argv_as_dict, parser_keyvalue_argv_as_dict, store_info_dict


def _is_input_consistant(argv_list: list):
    """
    判别输入是否一致(都为key value pair或者顺序调用)
    :param argv_list:
    :return:
    """
    keyvalue_list = [argv for argv in argv_list if '=' in argv]

    return (len(keyvalue_list) == 0) or (len(keyvalue_list) == len(argv_list))


def _is_keyvalue_pair(argv_list: list):
    """
    判别是key value pair还是顺序调用
    :param argv_list:
    :return:
    """
    return '=' in argv_list[0]


def parser_argv_to_info_dict(info_dict: InfoDict, file_pos: str, is_classify: bool):
    """
    解析输入参数存到info_dict中
    整体流程：
    1.判别输入是否一致(都为key value pair或者顺序调用)
    2.判别是key value pair还是顺序调用
    3.调用相应的解析
    :param info_dict:
    :param is_classify:
    :return:
    """
    argv_list = sys.argv[1:]
    assertor.greater_or_equal_assert(len(argv_list), 1)

    # 1.判别输入是否一致(都为key value pair或者顺序调用)
    if not _is_input_consistant(argv_list):
        raise ValueError('The input mix with keyvalue pair and none keyvalue pair parameter')
    # 2.判别是key value pair还是顺序调用
    if _is_keyvalue_pair(argv_list):
        # 3.调用相应的解析
        input_dict = parser_keyvalue_argv_as_dict(argv_list, file_pos)
    else:
        # 3.调用相应的解析
        input_dict = parser_argv_as_dict(argv_list, file_pos, is_classify)

    info_dict = store_info_dict(info_dict, input_dict)

    if 'full_body_classify' in file_pos:
        info_dict.alg_name = 'full_body_classify'
    else:
        info_dict.alg_name = file_pos.split('get_')[-1].replace('_contours.py', '')

    return info_dict
