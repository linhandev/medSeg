#encoding=utf-8
import numpy as np
import nibabel as nib
from tqdm import tqdm
from util import *
from config import *

'''
测试预处理代码，包含脚手架代码，保存成nii文件
写降噪和增强的代码
'''

volumes = listdir(volumes_path)
labels = listdir(labels_path)

if not os.path.exists(preprocess_path):
	os.makedirs(preprocess_path)

pbar=tqdm(range(len(volumes)) ,desc="数据处理中")
for i in range(len(volumes)):

	pbar.set_postfix(filename=volumes[i].rstrip(".nii"))
	pbar.update(1)

# 	print(volumes[i], labels[i])

	volf = nib.load(os.path.join(volumes_path, volumes[i]))
	labf = nib.load(os.path.join(labels_path, labels[i]))

	volume = volf.get_fdata()
	label = labf.get_fdata()

	volume=np.clip(volume,-1024,1024)
	label = clip_label(label, 1)

	if label.sum() < 32:
		continue

	# bb_min, bb_max = get_bbs(label)
	# label = crop_to_bbs(label, bb_min, bb_max)[0]
	# volume = crop_to_bbs(volume, bb_min, bb_max)[0]
	#
	# label = pad_volume(label, [512, 512, 700], 0)  # NOTE: 注意这里使用 0
	# volume = pad_volume(volume, [512, 512, 700], -1024)

	volume = volume.astype(np.float16)
	label = label.astype(np.int8)

	for frame in range(1,volume.shape[2]-1):
		if np.sum(label[:,:,frame]) > 4:
			vol=volume[:,:,frame-1:frame+2]
			lab=label[:,:,frame]
			lab=lab.reshape([lab.shape[0],lab.shape[1],1])

			vol=np.swapaxes(vol,0,2)
			lab=np.swapaxes(lab,0,2)  #[3,512,512],3 2 1 的顺序，用的时候倒回来, CWH

			data=np.concatenate((vol,lab),axis=0)
			file_name = "lits-{}-{}.npy".format(volumes[i].rstrip(".nii").lstrip("volume"),frame)
			np.save(os.path.join(preprocess_path, file_name), data )

pbar.close()
