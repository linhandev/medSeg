# coding=utf-8
"""
将一个目录下的所有文件和文件夹按照原来的文件结构打包，每个包不超过指定大小
"""

# TODO 测试压缩后删除原文件功能
# TODO: 修改成按照zip文件的大小显示进度条

import zipfile
import os
from tqdm import tqdm
import platform
import argparse
import logging


def get_args():
    parser = argparse.ArgumentParser("zip_dataset")
    parser.add_argument(
        "-i", "--dataset_dir", type=str, required=True, help="[必填] 需要压缩的数据集路径，所有文件所在的文件夹"
    )
    parser.add_argument(
        "-o",
        "--zip_dir",
        type=str,
        required=True,
        help="[必填] 压缩后的压缩包保存路径，如果有条件可以和待压缩数据放到不同硬件上，能加快一点速度",
    )
    parser.add_argument("--size", type=float, default=10.0, help="[可选] 压缩文件过程中每个包不超过这个大小，G为单位")
    parser.add_argument(
        "-m", "--method", type=str, default="zip", help="[可选] 压缩方法，可选的有:store(只打包不压缩),zip,bz2,lzma",
    )
    parser.add_argument(
        "-v", "--verbos", action="store_true", default=False, help="[可选] 执行过程中显示详细信息"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="[可选] 执行过程中显示详细信息")
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        default=False,
        help="[慎用] 在压缩完文件后删除对应的原文件，如果盘上空间不够压完就删掉原文件不会炸盘。！！慎用此功能！！",
    )

    return parser.parse_args()


def do_zip(args):
    # 1. 参数校验和设置
    if args.verbos:
        level = logging.INFO
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.CRITICAL
    logging.basicConfig(format="%(asctime)s : %(message)s", level=level)

    # TODO: 不同压缩方法不用后缀
    methods = {
        "zip": zipfile.ZIP_DEFLATED,
        "store": zipfile.ZIP_STORED,
        "bz2": zipfile.ZIP_BZIP2,
        "lzma": zipfile.ZIP_LZMA,
    }
    try:
        mode = methods[args.method]
    except KeyError:
        raise RuntimeError("mode 参数 {} 不合法".format(mode))

    # if args.method == "zip":
    #     mode = zipfile.ZIP_DEFLATED
    # elif args.method == "store":
    #     mode = zipfile.ZIP_STORED
    # elif args.method == "bz2":
    #     mode = zipfile.ZIP_BZIP2
    # elif argms.mode == "lzma":
    #     mode = zipfile.ZIP_LZMA
    # else:
    #     raise RuntimeError("mode参数 {} 不合法".format(mode))

    # 2. 确认路径和数据集名
    print("\n", os.listdir(args.dataset_dir)[:10], "\n")
    print("以上是您指定的待压缩路径 {} 下的前10个文件(夹)，请确定该路径是否正确".format(args.dataset_dir))
    cmd = input("如果 是 请输入 y/Y ，按其他任意键退出执行: ")
    if cmd != "y" and cmd != "Y":
        logging.error("用户退出执行")
        exit(0)

    if not os.path.exists(args.zip_dir):  # 如果zip输出路径不存在创建它
        logging.info("创建zip文件夹: {}".format(args.zip_dir))
        os.makedirs(args.zip_dir)

    dataset_name = os.path.basename(args.dataset_dir.rstrip("\\").rstrip("/"))  # 文件夹名做数据集名
    logging.info("默认数据集名称为： {}".format(dataset_name))
    print("默认数据集名称为： {}".format(dataset_name))
    cmd = input("确认使用该名称请输入 y/Y，如想使用其他名称请输入: ")
    if cmd != "y" and cmd != "Y":
        dataset_name = cmd
    logging.info("最终使用数据集名称为： {}".format(dataset_name))

    # 3. 制作当前压缩包名，创建压缩包文件
    zip_num = 1
    curr_name = "{}-{}.zip".format(dataset_name, zip_num)
    curr_zip_path = os.path.join(args.zip_dir, curr_name)
    f = zipfile.ZipFile(curr_zip_path, "a", mode)

    files_list = []  # 用来存储待压缩的文件路径和在压缩包中的路径，存到一定数量之后一起往包里压，避免频繁查看当前压缩包大小
    list_size = 0  # 当前 files_list 中文件总大小， 单位B
    zip_tot_size = args.size * 1000 ** 3  # 每个压缩包不超过这个大小，单位B。因为有的设备上是按照1k换算的，所以保险用1B*1000^3做1G
    zip_left_size = zip_tot_size  # 当前压缩包离最大大小还有多少

    """
    压缩的整体策略是
    1. 将文件路径添加进 files_list，直到列表中再添加一个文件就会超过压缩包离最大限制的空间 zip_left_size。
        因为files_list文件的文件计算的是没压缩的大小，所以这些文件都加进压缩包大小一定不会超过限制。
    2. 将 files_list 中的文件都加入压缩包，检查当前压缩包的大小，决定是否开新压缩包。 之后继续制作 files_list
    3. 经过多次 2 的添加压缩包大小会接近最大限制，开新压缩包的条件是当前压缩包的大小加上当前准备压缩的文件大小超过了最大限制。
        这里可能会有一点浪费，但是这个文件没实际压进包没法知道是不是会超限制，所以就直接保守开新包了。这里在压缩率比较高包还很小的时候可能会不停的开新包。
    """
    for dirpath, dirnames, filenames in os.walk(os.path.join(args.dataset_dir)):
        for filename in filenames:
            # 获取当前文件大小，判断列表中加入这个文件是否超过限制
            curr_file_size = os.path.getsize(os.path.join(dirpath, filename))
            logging.debug("Name: {}, Size: {}".format(filename, curr_file_size / 1024 ** 2))
            list_size += curr_file_size

            if list_size >= zip_left_size:  # 如果当前列表中未压缩文件的大小大于zip包能装的大小，那么开始压包
                logging.info("当前列表中文件大小是： {} M ".format(list_size / 1024 ** 2))
                logging.info("当前压缩包剩余大小： {} M".format(zip_left_size / 1024 ** 2))

                logging.critical("正在将 {} 个文件写入压缩包".format(len(files_list)))
                logging.debug("前三个文件是: {}".format(str(files_list[:3])))
                logging.debug("最后三个文件是: {}".format(str(files_list[-3:])))
                # 将列表里所有的文件写入zip
                for pair in tqdm(files_list, ascii=True):
                    f.write(pair[0], pair[1])
                    if args.delete:
                        os.remove(pair[0])

                files_list = []  # 写入完成，清空列表
                # 循环中这个pass的文件是在if之后加入列表的，所以列表文件的大小直接就是这个pass文件的大小，下面就添加了
                list_size = curr_file_size

                curr_zip_size = os.path.getsize(curr_zip_path)
                logging.info("当前压缩包的大小是: {} M\n".format(curr_zip_size / 1024 ** 2))

                if curr_zip_size + curr_file_size > zip_tot_size:  # 如果加入这个文件压缩包就超大小限制了就开新的压缩包
                    f.close()
                    zip_num += 1
                    curr_name = "{}-{}.zip".format(dataset_name, zip_num)
                    curr_zip_path = os.path.join(args.zip_dir, curr_name)
                    f = zipfile.ZipFile(curr_zip_path, "a", mode)
                    logging.critical("\n\n\n正创建新的压缩包: {} ".format(curr_name))
                    zip_left_size = zip_tot_size
                else:
                    zip_left_size = zip_tot_size - curr_zip_size

            # 第一个是文件路径，第二个是压缩包中的路径，压缩包中保存原来文件夹的结构
            files_list.append(
                [
                    os.path.join(dirpath, filename),
                    os.path.join(dataset_name, dirpath[len(args.dataset_dir) + 1 :], filename),
                ]
            )

    # 最后一个压缩包一般都不会到限制的大小触发写入，剩下的所有文件写入最后一个压缩包
    if len(files_list) != 0:
        logging.critical("正在将 {} 个文件写入最后一个压缩包".format(len(files_list)))
        for pair in tqdm(files_list, ascii=True):
            f.write(pair[0], pair[1])
        files_list = []
        list_size = 0
        f.close()

    logging.critical("压缩结束，共 {} 个压缩包".format(zip_num))


if __name__ == "__main__":
    args = get_args()
    do_zip(args)
