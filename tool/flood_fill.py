import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import flood_fill
import numpy as np
import nibabel as nib
import queue
import os

# TODO:  -60以下前景的要去掉

lab_path = '/home/lin/Desktop/data/ann/flood/'
fill_path = '/home/lin/Desktop/data/ann/flooded/'
processed = 0
lab_names = os.listdir(lab_path)
for lab_name in lab_names:
	print("--------")
	print(lab_name)

	if os.path.exists(os.path.join(fill_path, lab_name)):
		print("[skp]跳过已经手工refine过的标签")
		continue

	labf = nib.load(os.path.join(lab_path, lab_name) )
	lab = labf.get_fdata()

	# print(np.where(lab == 2))
	if len(np.where(lab == 2)[0] ) == 0:
		print("[skp]跳过没有lab2的标签")
		continue


	print(lab.shape)
	processed += 1

	# 第 2 个维度是层数，从0开始，和itk里面下表是 i 对应 i + 1
	# 片内是 itk 显示的片子顺时针转90度
	# 第 0 个维度在数组中是上到下，在itk中是左到右
	# 第 1 个维度在数组中是左到右，在itk中是上到下

	for sli_ind in range(lab.shape[2]):

		seeds = np.where(lab[:,:,sli_ind] == 2)
		# print(seeds)

		# plt.imshow(lab[:, :, sli_ind])
		# plt.show()

		for i in range(len(seeds[0])):
			lab[seeds[0][i], seeds[1][i], sli_ind] = 0
			slice = lab[:, :, sli_ind]
			flood_fill(slice, (seeds[1][i], seeds[0][i]), 1, selem=[[0,1,0],[1,0,1],[0,1,0]], inplace=True)

			lab[seeds[0][i], seeds[1][i], sli_ind] = 1

		# plt.imshow(lab[:, :, sli_ind])
		# plt.show()

	filled = nib.Nifti1Image(lab, np.eye(4))
	nib.save(filled, os.path.join(fill_path, lab_name) )
	print("\t处理完成")

print('共: ', len(lab_names), '处理了: ', processed)
