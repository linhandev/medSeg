# coding=utf-8

import os


def judge_mkdir(path: str):
    """
    判别位置,如果不存在则创建文件夹
    """
    # 去除首空格
    path = path.strip()
    # 判别是否存在路径,如果不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)

