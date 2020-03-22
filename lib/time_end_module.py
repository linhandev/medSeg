#coding=utf-8

import time
from lib.info_dict_module import InfoDict


def time_end(info_dict: InfoDict, time_name):
    info_dict.time_dict[time_name + '_end'] = time.time()
    if (time_name + '_diff' not in info_dict.time_dict.keys()) or (info_dict.time_dict[time_name + '_diff'] == -1.):
        info_dict.time_dict[time_name + '_diff'] = info_dict.time_dict[time_name + '_end'] - info_dict.time_dict[
            time_name + '_begin']
    else:
        info_dict.time_dict[time_name + '_diff'] += info_dict.time_dict[time_name + '_end'] - info_dict.time_dict[
            time_name + '_begin']
