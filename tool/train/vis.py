import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt


for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]


base_dir = "/home/lin/Desktop/data/lits/img"
img_dir = osp.join(base_dir, "JPEGImages")
lab_dir = osp.join(base_dir, "Annotations")

for f in os.listdir(lab_dir)[:30]:
    img = osp.join(img_dir, f)
    lab = osp.join(lab_dir, f)

    img = cv2.imread(img)
    lab = cv2.imread(lab, cv2.IMREAD_UNCHANGED)
    print(lab.sum())
    lab = lab * 255
    # lab = lab.reshape([512, 512, 1])
    print(img.shape, lab.shape)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(lab)
    plt.show()
