import six
from ast import literal_eval
import codecs
import yaml

# 使用的时候如果直接赋值出去，默认是不可变的，如果需要再赋值一定注意
class PjConfig(dict):
    def __init__(self, *args, **kwargs):
        super(PjConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = PjConfig()
        return self[key]

    def __setitem__(self, key, value):
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but PjConfig is immutable'.format(key, value)
            )
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(PjConfig, self).__setitem__(key, value)

    def update_from_Config(self, other):
        if isinstance(other, dict):
            other = PjConfig(other)
        assert isinstance(other, PjConfig)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError("Non-existent config key: {}".format(key))

    # def check_and_infer(self):
    #     if self.DATASET.IMAGE_TYPE in ["rgb", "gray"]:
    #         self.DATASET.DATA_DIM = 3
    #     elif self.DATASET.IMAGE_TYPE in ["rgba"]:
    #         self.DATASET.DATA_DIM = 4
    #     else:
    #         raise KeyError(
    #             "DATASET.IMAGE_TYPE config error, only support `rgb`, `gray` and `rgba`"
    #         )
    #     if self.MEAN is not None:
    #         self.DATASET.PADDING_VALUE = [x * 255.0 for x in self.MEAN]
    #
    #     if not self.TRAIN_CROP_SIZE:
    #         raise ValueError(
    #             "TRAIN_CROP_SIZE is empty! Please set a pair of values in format (width, height)"
    #         )
    #
    #     if not self.EVAL_CROP_SIZE:
    #         raise ValueError(
    #             "EVAL_CROP_SIZE is empty! Please set a pair of values in format (width, height)"
    #         )
    #
    #     # Ensure file list is use UTF-8 encoding
    #     train_sets = codecs.open(self.DATASET.TRAIN_FILE_LIST, "r", "utf-8").readlines()
    #     val_sets = codecs.open(self.DATASET.VAL_FILE_LIST, "r", "utf-8").readlines()
    #     test_sets = codecs.open(self.DATASET.TEST_FILE_LIST, "r", "utf-8").readlines()
    #     self.DATASET.TRAIN_TOTAL_IMAGES = len(train_sets)
    #     self.DATASET.VAL_TOTAL_IMAGES = len(val_sets)
    #     self.DATASET.TEST_TOTAL_IMAGES = len(test_sets)
    #
    #     if self.MODEL.MODEL_NAME == "icnet" and len(self.MODEL.MULTI_LOSS_WEIGHT) != 3:
    #         self.MODEL.MULTI_LOSS_WEIGHT = [1.0, 0.4, 0.16]
    #
    #     if self.AUG.AUG_METHOD not in ["unpadding", "stepscaling", "rangescaling"]:
    #         raise ValueError(
    #             "AUG.AUG_METHOD config error, only support `unpadding`, `unpadding` and `rangescaling`"
    #         )

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".format(config_list)
            )
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError("Non-existent config key: {}".format(key))

    def update_from_file(self, config_file):
        with codecs.open(config_file, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        self.update_from_Config(dic)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, PjConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable


cfg = PjConfig()

"""数据集配置"""
# 数据集名称
cfg.DATA.NAME = "lits"
cfg.DATA.VOLUMES_PATH = "/home/aistudio/data/volume"
cfg.DATA.LABELS_PATH = "/home/aistudio/data/label"
cfg.DATA.PREP_PATH = "/home/aistudio/data/preprocess"
# z 方向初始化可以指定一个独立的输出文件路径
cfg.DATA.Z_PREP_PATH = cfg.DATA.PREP_PATH
cfg.DATA.SUMMARY_FILE = "./{}.csv".format(cfg.DATA.NAME)

""" 预处理配置 """
cfg.PREP.PLANE = "xy"  # 预处理进行的平面
# 处理过程中所有比这个数字大的标签都设为前景
cfg.PREP.FRONT = 1
# 是否将数据只 crop 到前景
cfg.PREP.CROP = False
# 是否对数据插值改变大小
cfg.PREP.INTERP = False
# 进行插值的话目标片间间隔是多少，单位mm，-1的维度不会进行插值
cfg.PREP.INTERP_PIXDIM = (-1, -1, 1.0)
cfg.PREP.WINDOW = False  # 是否进行窗口化
cfg.PREP.WWWC = (180, 50)  # 窗宽窗位
# 丢弃前景数量少于thresh的slice
cfg.PREP.THRESH = 256
# 3D的数据在开始切割之前pad到这个大小，-1的维度会放着不动
cfg.PREP.SIZE = (512, 512, -1)
# 预处理过程中多少组数据组成一个npz文件
cfg.PREP.BATCH_SIZE = 128  # 可以先跑bs=1，看看一对数据多大

"""训练配置"""
cfg.TRAIN.DATA_PATH = "/home/aistudio/data/preprocess"
# 训练数据的数量，用来显示训练进度条和时间估计，如果不知道有多少写-1
cfg.TRAIN.DATA_COUNT = -1
cfg.TRAIN.PRETRAINED_WEIGHT = ""
cfg.TRAIN.INF_MODEL_PATH = "./model/lits/inf"
cfg.TRAIN.CKPT_MODEL_PATH = "./model/lits/ckpt"
cfg.TRAIN.BEST_MODEL_PATH = "./model/lits/best"

cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.EPOCHS = 20
cfg.TRAIN.ARCHITECTURE = "unet_base"

cfg.TRAIN.USE_GPU = True
# 进行验证
cfg.TRAIN.DO_EVAL = True
# 每 snapchost_epoch做一次eval并保存模型
cfg.TRAIN.SNAPSHOT_BATCH = 1000
# VDL log路径
cfg.TRAIN.VDL_LOG = "/home/aistudio/log"

"""数据增强"""
cfg.AUG.WINDOWLIZE = True
cfg.AUG.WWWC = cfg.PREP.WWWC
# 不单独为增强操作设做不做的config，不想做概率设成 0
# 注意CWH
# 每个维度进行翻转增强的概率
cfg.AUG.FLIP.RATIO = (0, 0, 0.5)
# 进行旋转增强的概率
cfg.AUG.ROTATE.RATIO = (0, 0.5, 0)
# 旋转的角度范围，单位度
cfg.AUG.ROTATE.RANGE = (0, (-15, 15), 0)
# 进行缩放的概率
cfg.AUG.ZOOM.RATIO = (0, 0.3, 0.3)
# 进行缩放的比例
cfg.AUG.ZOOM.RANGE = ((1, 1), (0.8, 1), (0.8, 1))
# 进行随机crop的目标大小
cfg.AUG.CROP.SIZE = (3, 512, 512)

"""推理配置"""
# 推理的输入数据路径
cfg.INFER.PATH.INPUT = "/home/aistudio/data/inference"
# 推理的结果输出路径
cfg.INFER.PATH.OUTPUT = "/home/aistudio/data/infer_lab"
# 推理的模型权重路径
cfg.INFER.PATH.PARAM = "/home/aistudio/weight/liver/inf"

# 是否使用GPU进行推理
cfg.INFER.USE_GPU = False
# 推理过程中的 batch_size
cfg.INFER.BATCH_SIZE = 128
# 是否进行窗口化，这个和训练过程中的配置应当相同
cfg.INFER.WINDOWLIZE = True
# 窗宽窗位
cfg.INFER.WWWC = (180, 50)
# 是否进行插值
cfg.INFER.DO_INTERP = False
# 如果进行插值，目标的spacing，-1的维度忽略
cfg.INFER.SPACING = [-1, -1, 1]
# 是否进行最大连通块过滤
cfg.INFER.FILTER_LARGES = True
# 推理过程中区分前景和背景的阈值
cfg.INFER.THRESH = 0.5

""" 测试配置 """
# 分割结果的路径
cfg.EVAL.PATH.RESULT = "/home/aistudio/data/infer_lab"
# 分割GT标签的路径
cfg.EVAL.PATH.GT = "/home/aistudio/data/eval_lab"
# 测试过程中要计算的指标，包括
# FP，FN，TP，TN(绝对数量)
# Precision,Recall/Sensitivity,Specificity,Accuracy,Kappa
# Dice,IOU/VOE
cfg.EVAL.METRICS = ["IOU"]

# print(cfg.EVAL.METRICS)
