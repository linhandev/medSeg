#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: store_data_info.py
@Time: 2018-12-04 13:21
@Last_update: 2018-12-04 13:21
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from lib.info_dict_module import InfoDict
import lib.assert_module as assertor


def store_info_dict(info_dict=InfoDict(), input_dict={}, **kwargs):
    assertor.type_assert(input_dict, dict)
    assertor.type_assert(kwargs, dict)

    for temp_dict_key, temp_dict_value in input_dict.items():
        info_dict[temp_dict_key] = temp_dict_value


    for temp_dict_key, temp_dict_value in kwargs.items():
        info_dict[temp_dict_key] = temp_dict_value


    return info_dict
