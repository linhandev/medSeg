# coding=utf-8

from lib.time_type_module import TimeType
from lib.time_begin_module import time_begin
from lib.time_end_module import time_end
from lib.dcm2array_basis import load_dcm_scan

def load_data(info_dict):
    """
    加载dicom数据，获得患者基本信息，以及CT图像
    :return: 椎体中心点3D坐标，N*4 第一列椎体类型（序号），第2-4列为椎体中心的z y x 坐标
    """
    print('\nBegin loading data')
    time_begin(info_dict, TimeType.load_data)

    image_raw, info_dict = load_dcm_scan(info_dict)


    time_end(info_dict, TimeType.load_data)
    print('Done loading data, runtime: %.3f\n'
          % info_dict.time_dict['load_data_diff'])

    return image_raw, info_dict


if __name__ == '__main__':
 print('Lianxin')
