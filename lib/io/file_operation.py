# coding=utf-8
# @Time	  : 2018-11-21 15:04
# @Author   : Monolith
# @FileName : file_operation.py
# @License  : (C)LINKINGMED,2018
# @Contact  : baibaidj@126.com, 18600369643

# from concurrent.futures import ProcessPoolExecutor

# 扫描目标文件夹

def is_dcm_file(file):
    """
    # dcm file could end with an extention as well as no extension at all
    :param file: 文件路径
    :return: True or False
    """

    inc_extension = (r'.' not in file[-5:]) or ('.dcm' in file)
    exc_extension = ('.json' in file) or ('.csv' in file)
    return inc_extension and not exc_extension