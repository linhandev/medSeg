# 整个项目的设置文件，包含各种路径，运行模式和一些参数
import os

# 所有的路径都要 / 封死
data_base_dir = "/home/aistudio/data/"  # 数据基路径
code_base_dir = "/home/aistudio/work/"  # 代码基路径，就是项目所在的路径

# 训练数据路径
volumes_path = os.path.join(data_base_dir, "volume")  # 体数据路径
labels_path = os.path.join(data_base_dir, "label")  # 标签路径
preprocess_path = os.path.join(data_base_dir, "preprocess")  # 预处理生成的npy数据存储路径
z_prep_path = os.path.join(data_base_dir, "zprep")  # Z 方向预处理生成的npy存储路径

# 推理数据路径
inference_path = os.path.join(data_base_dir, "inference")  # 做前向的体数据路径
inference_label_path = os.path.join(data_base_dir, "inf_lab/")  # 如果做测试，前向的标签放在这

# 权重路径
# infer_param_path = "/home/aistudio/weights/liver/tumor_inf"
# infer_param_path = "/home/aistudio/work/params/unet_base/inf"

# BML
# preprocess_path = './train_data'


# 预处理数据转换png路径
# volumes_png_path = data_base_dir+"vol_png/"
# labels_png_path = data_base_dir+"lab_png/"

# 统计量路径
# vol_plt_path=data_base_dir+"vol_plt/"       #存预处理数据分布图片的路径
# lab_plt_path=data_base_dir+"lab_plt/"   #存插值之后label的分布
# vol_percentage_path=data_base_dir+"vol_percentage/" #存 volume 数据分布txt文件的路径
# lab_percentage_path=data_base_dir+"lab_percentage/" #存 label 数据分布txt文件的路径
