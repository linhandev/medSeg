# encoding=utf-8
# 在Z方向进行预处理，做病人侧面看的视图

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm
from config import *
from util import *

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

	header = volf.header.structarr
	save_info(volumes[i], header, 'vol_info.csv')
	spacing = [1, 1, 1]
	pixdim = [header['pixdim'][1], header['pixdim'][2], header['pixdim'][3]]  # pixdim 是这张 ct 三个维度的间距
	ratio = [pixdim[0]/spacing[0], pixdim[1]/spacing[1], pixdim[2]/spacing[2]]
	ratio = [1, 1, ratio[2]]
	volume = volf.get_fdata()
	label = labf.get_fdata()

	volume=np.clip(volume,-1024,1024)
	label = clip_label(label, 1)

	volume=scipy.ndimage.interpolation.zoom(volume,ratio,order=3)
	label=scipy.ndimage.interpolation.zoom(label,ratio,order=0)

	# for ind in range(512):
	# 	plt.imshow(volume[ind, :, :])
	# 	plt.show()
	# 	plt.close()


	if label.sum() < 32:
		continue

	# bb_min, bb_max = get_bbs(label)
	# label = crop_to_bbs(label, bb_min, bb_max)[0]
	# volume = crop_to_bbs(volume, bb_min, bb_max)[0]
	#
	label = pad_volume(label, 512, 0)  # NOTE: 注意这里使用 0
	volume = pad_volume(volume, 512, -1024)
	print(label.shape)
	volume = volume.astype(np.float16)
	label = label.astype(np.int8)

	for frame in range(1,volume.shape[0]-1):
		if np.sum(label[frame ,: ,:]) > 32:
			vol=volume[frame-1:frame+2, :, :]
			lab=label[frame ,: ,:]
			lab = lab.reshape([lab.shape[0],lab.shape[1],1])

			# vol=np.swapaxes(vol,0,2)
			lab=np.swapaxes(lab,0,2)  #[3,512,512],3 2 1 的顺序，用的时候倒回来, CWH

			# print(vol.shape)
			# print(lab.shape)

			data=np.concatenate((vol,lab),axis=0)
			file_name = "lits{}-{}.npy".format(volumes[i].rstrip(".nii").lstrip("volume"),frame)
			np.save(os.path.join(preprocess_path, file_name),data )

pbar.close()
