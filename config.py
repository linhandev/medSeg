# 整个算法的设置文件，包含各种路径，运行模式和一些参数
import os 
#所有的路径都要 / 封死
data_base_dir="/home/aistudio/data/"
code_base_dir="/home/aistudio/work/"

# 训练数据路径
volumes_path = data_base_dir+"volume/"  # 体数据路径
labels_path = data_base_dir+"label/"    #标签路径
preprocess_path = data_base_dir+"preprocess/" #生成的tfrecord路径


# 预处理数据转换png路径
volumes_png_path = data_base_dir+"vol_png/"
labels_png_path = data_base_dir+"lab_png/"

# 统计量路径
vol_plt_path=data_base_dir+"vol_plt/"       #存预处理数据分布图片的路径
lab_plt_path=data_base_dir+"lab_plt/"   #存插值之后label的分布
vol_percentage_path=data_base_dir+"vol_percentage/" #存 volume 数据分布txt文件的路径
lab_percentage_path=data_base_dir+"lab_percentage/" #存 label 数据分布txt文件的路径


inference_path = data_base_dir+"inference/"  # 做前向的体数据路径
inference_label_path = data_base_dir+"inf_lab/" # 如果做测试，前向的标签放在这



# 预处理 对每个人生成两个npy，切放到训练的时候去做，能给训练更大的灵活性
pixdims=[1,1,1] # 默认间隔都搞成1mm
norms=[0,0,0] # ct强度的最大值，最小值，中位数

plt_permutation=False  #生成统计量加深对数据集理解，会托慢速度需要创建路径


# 训练超参数
batch_size=2
epoch=10
pre_load_batch=1

patch_size=[128,128,128]
stride=[64,64,64]


# isExists=os.path.exists(path)
# if not isExists:
# 	os.makedirs(path)
