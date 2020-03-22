#coding=utf-8

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
