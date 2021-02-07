#utf-8
'''
扫描数据集，生成summary
'''

from util import *
import nibabel as nib
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

volumes=listdir(volumes_path)
labels=listdir(labels_path)
assert len(volumes) == len(labels), '训练中体数据和标签数量不相等'

print("数据集中共{}样本".format(len(volumes)) )

pixdims=[]
shapes=[]
norms=[]

pbar=tqdm(range(len(volumes)) ,desc="正在统计")
for i in range(len(volumes)):
	pbar.set_postfix(filename=volumes[i].rstrip(".nii"))
	pbar.update(1)

	volf=nib.load(os.path.join(volumes_path,volumes[i]))
	labf=nib.load(os.path.join(labels_path,labels[i]))

	header=volf.header.structarr
	shape=volf.header.get_data_shape()
	shapes.append([shape[0],shape[1],shape[2]])
	pixdims.append([header['pixdim'][1],header['pixdim'][2],header['pixdim'][3]])

	volume=volf.get_fdata()
	norms.append([np.min(volume),np.median(volume),np.max(volume)])

	if plt_permutation:
		volume=np.reshape(volume,[volume.shape[0]*volume.shape[1]*volume.shape[2]]) #change into 1d array to conduct analasys

		plt.title("{} [{},{}]".format(volumes[i].rstrip(".nii"),np.min(volume),np.max(volume)))
		plt.xlabel("size:[{},{},{}] pixdims:[{},{},{}] ".format(shape[0],shape[1],shape[2],header['pixdim'][1],header['pixdim'][2],header['pixdim'][3] ) )
		nums,bins,patchs=plt.hist(volume,bins=1000)
		plt.savefig(vol_plt_path+volumes[i].rstrip(".nii")+".png")
		plt.close()

		file=open(os.path.join(vol_percentage_path,"{}.txt".format(volumes[i].rstrip(".nii")) ),"w")
		print("--------- {} --------".format(volumes[i]),file=file)

		sum=0
		for num in nums:
			sum+=num
		nowsum=0
		for j in range(0,len(nums)):
			nowsum+=nums[j]
		print("[{:<10f},{:<10f}] : {:>10} percentage : {}".format(bins[j],bins[j+1],nums[j],nowsum/sum),file=file)

		label=labf.get_fdata()
		label=np.reshape(label,[label.shape[0]*label.shape[1]*label.shape[2]])
		plt.title("{} [{},{}]".format(volumes[i].rstrip(".nii"),np.min(label),np.max(label)))
		plt.xlabel("size:[{},{},{}] pixdims:[{},{},{}] ".format(shape[0],shape[1],shape[2],header['pixdim'][1],header['pixdim'][2],header['pixdim'][3] ) )
		nums,bins,patchs=plt.hist(label,bins=3)
		plt.savefig(lab_plt_path+volumes[i].rstrip(".nii")+".png")
		plt.close()

		file=open(os.path.join(lab_percentage_path,"{}.txt".format(volumes[i].rstrip(".nii")) ),"w")
		print("--------- {} --------".format(volumes[i]),file=file)

		sum=0
		for num in nums:
			sum+=num
		nowsum=0
		for i in range(0,len(nums)):
			nowsum+=nums[i]
		print("[{:<10f},{:<10f}] : {:>10} percentage : {}".format(bins[i],bins[i+1],nums[i],nowsum/sum),file=file)

pbar.close()

spacing=np.median(pixdims,axis=0)
size=np.median(shapes,axis=0)

norm=[]
norm.append(np.min(norms[:][0]))
norm.append(np.median(norms[:][1]))
norm.append(np.max(norms[:][2]))

print(spacing,size,norm)

file=open("./summary.txt","w")

print("spacing",file=file)
for dat in spacing:
	print(dat,file=file)

print("\nsize",file=file)
for dat in size:
	print(dat,file=file)

print("\nnorm",file=file)
for dat in norm:
	print(dat,file=file)
