import os

base_path = ""
for name in os.listdir(base_path):
    for seq in os.listdir(os.path.join(base_path, name)):
        dcm_path = os.path.join(base_path, name, seq)
        print(dcm_path)
        cmd = "dcm2niix -f {} -o /home/lin/Desktop/data/aorta/农安/nii_raw -c {} {}".format(
            name + seq, name + seq, dcm_path
        )
        print(cmd)
        os.system(cmd)
