import os

data_base_dir = "/home/aistudio/data/imgs"
scan_folder = "imgs"
label_folder = "labs"
txt_path = "/home/aistudio/data/"

split = [0, 0.7, 0.9, 1.0]  # 训练，验证和测试集的划分比例为 7:2:1
list_names = ["train_list.txt", "val_list.txt", "test_list.txt"]
curr_type = 0

img_count = len(os.listdir(os.path.join(data_base_dir, scan_folder)))
split = [int(x * img_count) for x in split]

f = open(os.path.join(txt_path, list_names[curr_type]), "w")
for ind, slice_name in enumerate(os.listdir(os.path.join(data_base_dir, scan_folder))):
    if ind < img_count - 1 and ind == split[curr_type + 1]:
        curr_type += 1
        f.close()
        f = open(os.path.join(txt_path, list_names[curr_type]), "w")
    print(
        "{}|{}".format(
            os.path.join(scan_folder, slice_name), os.path.join(label_folder, slice_name)
        ),
        file=f,
    )
f.close()
