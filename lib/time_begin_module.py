#coding=utf-8

import time
from lib.info_dict_module import InfoDict


def time_begin(info_dict: InfoDict, time_name):
    info_dict.time_dict[time_name+'_begin'] = time.time()
