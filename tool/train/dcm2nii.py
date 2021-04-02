import os
from util import to_pinyin

base_path = "/home/lin/Desktop/aorta/dcm/extern/"
for name in os.listdir(base_path):
    for seq in os.listdir(os.path.join(base_path, name)):
        dcm_path = os.path.join(base_path, name, seq)
        print(dcm_path)
        cmd = "dcm2niix -f {} -o /home/lin/Desktop/aorta/nii/raw/extern/ -c {} {}".format(
            "%n-%s", name + seq, dcm_path
        )
        print(cmd)
        os.system(cmd)
