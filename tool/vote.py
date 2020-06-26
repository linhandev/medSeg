import nibabel as nib
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import os


def listdir(path):
    dirs = os.listdir(path)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    if "checkpoint" in dirs:
        dirs.remove("checkpoint")

    dirs.sort()  # 通过一样的sort保持vol和seg的对应
    return dirs


voters_base = "/home/aistudio/data/voting"
voter_paths = ["voter1", "voter2", "voter3"]
voter_paths = [os.path.join(voters_base, path) for path in voter_paths]


def voting(fname):
    voter0 = nib.load(os.path.join(voter_paths[0], fname))
    merged = voter0.get_fdata()
    print("voting {}".format(fname))
    # print(merged.sum())
    for ind in range(1, len(voter_paths)):
        voterf = nib.load(os.path.join(voter_paths[ind], fname))
        merged += voterf.get_fdata()
    merged[merged > len(voter_paths) / 2] = 1
    merged[merged != 1] = 0


def main():
    voter_names = [listdir(path) for path in voter_paths]
    print(voter_names)
    for data_ind in range(len(voter_names[0])):
        for voter_ind in range(1, len(voter_names)):
            # print(data_ind, voter_ind)
            assert voter_names[voter_ind][data_ind] == voter_names[0][data_ind], "第 {} 组数据，{} 和 {} 名称不相同".format(
                data_ind, voter_names[voter_ind][data_ind], voter_names[0][data_ind]
            )
        # for fname in voter_names[0]:
    # voting(fname)
    with Pool(cpu_count()) as p:
        p.map(voting, voter_names[0])


if __name__ == "__main__":
    main()
