# 原来的数据和标签分别在一个目录里，进行随机split之后按照pdseg的目录结构放
import os
import shutil
import random

# shutil.move("/home/lin/Desktop/a/test", "/home/lin/Desktop/b")  # 递归移动


def listdir(path):
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")

    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs


def mv(curr, dest):
    print(os.path.join(dest.rstrip(dest.split("/")[-1])))
    if not os.path.exists(os.path.join(dest.rstrip(dest.split("/")[-1]))):
        os.makedirs(os.path.join(dest.rstrip(dest.split("/")[-1])))
    os.rename(curr, dest)


split = [8, 2, 0]  # train/val/test
base_dir = "/home/lin/Desktop/data/aorta/dataset/"
folders = ["images", "annotations"]
sub_folders = ["train", "val", "test"]
if not os.path.exists(base_dir):
    for fd1 in folders:
        for fd2 in sub_folders:
            os.makedirs(os.path.join(base_dir, fd1, fd2))

img_folder = "/home/lin/Desktop/data/aorta/dataset/scan/"
lab_folder = "/home/lin/Desktop/data/aorta/dataset/label/"

img_names = listdir(img_folder)
lab_names = listdir(lab_folder)

print(img_names[:20])
print("\n\n\n")
print(lab_names[:20])
split.insert(0, 0)
for ind in range(1, len(split)):
    split[ind] += split[ind - 1]

split = [x / split[-1] for x in split]
split = [int(len(img_names) * split[ind]) for ind in range(4)]
print(split)
for part in range(3):
    print("Moving to folder {}".format(sub_folders[part]))
    for ind in range(split[part], split[part + 1]):
        print(img_names[ind], lab_names[ind])
        assert img_names[ind] == lab_names[ind], "图片和标签名字对不上{}, {}".format(img_names[ind], lab_names[ind])
        mv(
            os.path.join(img_folder, img_names[ind]),
            os.path.join(base_dir, folders[0], sub_folders[part], img_names[ind]),
        )
        mv(
            os.path.join(lab_folder, lab_names[ind]),
            os.path.join(base_dir, folders[1], sub_folders[part], lab_names[ind]),
        )
