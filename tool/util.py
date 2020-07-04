import os


def listdir(path):
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")
    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs
