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
                'Attempted to set "{}" to "{}", but PjConfig is immutable'.format(
                    key, value
                )
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
        self.check()

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".format(
                    config_list
                )
            )
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError("Non-existent config key: {}".format(key))
        self.check()

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

    def check(self):
        if cfg.PREP.THICKNESS % 2 != 1:
            raise ValueError("2.5D预处理厚度 {} 不是奇数".format(cfg.TRAIN.THICKNESS))


cfg = PjConfig()

"""数据集配置"""
# 数据集名称
cfg.DATA.NAME = "lits"
# 输入的2D或3D图像路径
cfg.DATA.INPUTS_PATH = "/home/aistudio/data/scan"
# 标签路径
cfg.DATA.LABELS_PATH = "/home/aistudio/data/label"
# 预处理输出npz路径
cfg.DATA.PREP_PATH = "/home/aistudio/data/preprocess"
# z 方向初始化可以指定一个独立的输出文件路径
cfg.DATA.Z_PREP_PATH = cfg.DATA.PREP_PATH
# 预处理过程中数据信息写到这个文件
cfg.DATA.SUMMARY_FILE = "./{}.csv".format(cfg.DATA.NAME)

""" 预处理配置 """
# 预处理进行的平面
cfg.PREP.PLANE = "xy"
# 处理过程中所有比这个数字大的标签都设为前景
cfg.PREP.FRONT = 1
# 是否将数据只 crop 到前景
cfg.PREP.CROP = False
# 是否对数据插值改变大小
cfg.PREP.INTERP = False
# 进行插值的话目标片间间隔是多少，单位mm，-1的维度不会进行插值
cfg.PREP.INTERP_PIXDIM = (-1, -1, 1.0)
# 是否进行窗口化，在预处理阶段不建议做，灵活性太低
cfg.PREP.WINDOW = False
# 窗宽窗位
cfg.PREP.WWWC = (400, 0)
# 丢弃前景数量少于thresh的slice
cfg.PREP.THRESH = 256
# 3D的数据在开始切割之前pad到这个大小，-1的维度会放着不动
cfg.PREP.SIZE = (512, 512, -1)
# 2.5D预处理一片的厚度
cfg.PREP.THICKNESS = 3
# 预处理过程中多少组数据组成一个npz文件
# 可以先跑bs=1，看看一对数据多大；尽量至少将训练数据分入10个npz，否则分训练和验证集的时候会很不准
# 这个值不建议给成 2^n，这样更利于随机打乱数据
cfg.PREP.BATCH_SIZE = 128

"""训练配置"""
cfg.TRAIN.DATA_PATH = "/home/aistudio/data/preprocess"
# 训练数据的数量，用来显示训练进度条和时间估计，如果不知道有多少写-1
cfg.TRAIN.DATA_COUNT = -1
# 预训练权重路径，如果没有写空，有的话会尝试加载
cfg.TRAIN.PRETRAINED_WEIGHT = ""
# 预测裁剪模型保存路径
cfg.TRAIN.INF_MODEL_PATH = "./model/lits/inf"
# 可以继续训练的ckpt模型保存路径
cfg.TRAIN.CKPT_MODEL_PATH = "./model/lits/ckpt"
# 效果最好的模型保存路径
cfg.TRAIN.BEST_MODEL_PATH = "./model/lits/best"
# 训练过程中输入图像大小，不加channel
cfg.TRAIN.INPUT_SIZE = (512, 512)
# 训练过程中用的batch_size
cfg.TRAIN.BATCH_SIZE = 32
# 共训练多少个epoch
cfg.TRAIN.EPOCHS = 20
# 使用的模型结构
cfg.TRAIN.ARCHITECTURE = "res_unet"
# 使用的正则化方法，支持L1，L2，其他一切值都是不加正则化
cfg.TRAIN.REG_TYPE = "L1"
# 正则化的权重
cfg.TRAIN.REG_COEFF = 1e-6
# 梯度下降方法
cfg.TRAIN.OPTIMIZER = "adam"
# 学习率
cfg.TRAIN.LR = [0.003, 0.002, 0.001]
# 学习率变化step
cfg.TRAIN.BOUNDARIES = [10000, 20000]
# Loss 支持ce，dice，miou，wce，focal
cfg.TRAIN.LOSS = ["ce", "dice"]
# 是否使用GPU进行训练
cfg.TRAIN.USE_GPU = False
# 进行验证
cfg.TRAIN.DO_EVAL = False
# 每 snapchost_epoch 做一次eval并保存模型
cfg.TRAIN.SNAPSHOT_BATCH = 500
# 每 disp_epoch 打出一次训练过程
cfg.TRAIN.DISP_BATCH = 10
# VDL log路径
cfg.TRAIN.VDL_LOG = "/home/aistudio/log"

""" HRNET 设置"""
# HRNET STAGE2 设置
cfg.MODEL.HRNET.STAGE2.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS = [40, 80]
# HRNET STAGE3 设置
cfg.MODEL.HRNET.STAGE3.NUM_MODULES = 4
cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS = [40, 80, 160]
# HRNET STAGE4 设置
cfg.MODEL.HRNET.STAGE4.NUM_MODULES = 3
cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS = [40, 80, 160, 320]

"""数据增强"""
# 不单独为增强操作设做不做的config，不想做概率设成 0，注意CWH
# 每个维度进行翻转增强的概率，CWH
# 是否进行窗口化
cfg.AUG.WINDOWLIZE = True
# 窗宽窗位
cfg.AUG.WWWC = cfg.PREP.WWWC
# 进行翻转数据增强的概率
cfg.AUG.FLIP.RATIO = (0, 0, 0)
# 进行旋转增强的概率
cfg.AUG.ROTATE.RATIO = (0, 0, 0)
# 旋转的角度范围，单位度
cfg.AUG.ROTATE.RANGE = (0, (0, 0), 0)
# 进行缩放的概率
cfg.AUG.ZOOM.RATIO = (0, 0, 0)
# 进行缩放的比例
cfg.AUG.ZOOM.RANGE = ((1, 1), (1, 1), (1, 1))
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
cfg.INFER.WWWC = cfg.PREP.WWWC
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
cfg.EVAL.PATH.SEG = "/home/aistudio/data/infer_lab"
# 分割GT标签的路径
cfg.EVAL.PATH.GT = "/home/aistudio/data/eval_lab"
# 评估结果存储的文件
cfg.EVAL.PATH.NAME = "eval"
# 测试过程中要计算的指标，包括
# FP，FN，TP，TN(绝对数量)
# Precision,Recall/Sensitivity,Specificity,Accuracy,Kappa
# Dice,IOU/VOE
cfg.EVAL.METRICS = [
    "IOU",
    "Dice",
    "TP",
    "TN",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "Accuracy",
]
