# coding=utf-8
# @Time	  : 2019-01-14 16:22
# @Author   : Monolith
# @FileName : h5_rw.py
# @License  : (C)LINKINGMED,2018
# @Contact  : baibaidj@126.com, 18600369643
import os
import h5py
import csv
import numpy as np


def save_dict_as_hdf5(
        save_file_dir: str, save_file_name: str,
        input_dict: dict, useCompression=False) -> None:
    """
    保存dict为hdf5文件
    :param save_file_dir: str
    :param save_file_name: str
    :param input_dict: dict
    :param useCompression: bool
    :return: None
    """
    # 断言保证
    # assertor.path_exist_assert(save_file_dir)

    with h5py.File(os.path.join(save_file_dir, save_file_name), 'w') as f:
        for key, value in input_dict.items():
            if useCompression:
                f.create_dataset(key, data=value, compression='gzip')
            else:
                f.create_dataset(key, data=value)


def get_hdf5_as_dict(*paths) -> dict:
    """
    通过hdf5文件读取,返回dict
    :param paths: tuple
    :return: data_dict: dict
    """
    # 断言保证
    # assertor.tuple_type_assert(paths, str)

    file_path = os.path.join(*paths)
    # 断言保证
    # assertor.file_exist_assert(file_path)

    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        for key, value in f.items():
            data_dict[key] = value.value
    return data_dict


def transfer_npy_to_hdf5(
        npy_file_path: str, save_file_dir: str,
        save_file_name: str, key_name: str) -> None:
    """
    读取npy文件, 转换为hdf5文件保存
    :param npy_file_path: str
    :param save_file_dir: str
    :param save_file_name: str
    :param key_name: str
    :return: None
    """
    # 断言保证
    # assertor.file_exist_assert(npy_file_path)
    # assertor.path_exist_assert(save_file_dir)
    # 读取npy数据
    npy_data = np.load(npy_file_path)
    # 变为dict
    rst_dict = {key_name: npy_data}
    # 存储为hdf5
    save_dict_as_hdf5(save_file_dir, save_file_name, rst_dict)


def save_list_to_str(input_list, save_dir, save_name):
    rst_str = '\n'.join(input_list)

    with open(os.path.join(save_dir, save_name), 'w') as f:
        f.write(rst_str)



def vector2csv(vector_list, save_dir, name):
    import pandas as pd
    vector_dict = {name: []}
    vector_dict[name] = vector_list
    df = pd.DataFrame(vector_dict)
    df.to_csv(os.path.join(save_dir, name + '.csv'), index=False)


def save2csv(matrix, save_path, file_name):
    with open(os.path.join(save_path, file_name), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # 将患者的CT层数、类别总数、每类层数写入CSV###
        for i in range(matrix.shape[0]):
            this_row = list(matrix[i])
            spamwriter.writerow(this_row)