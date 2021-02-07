from utils.config import cfg
from models.res_unet import res_unet
from models.unet_plain import unet_plain
from models.hrnet import hrnet
from models.deeplabv3p import deeplabv3p

# models = {
#     "unet_simple": unet_simple(volume, 2, [512, 512]),
#     "res_unet": unet_base(volume, 2, [512, 512]),
#     "deeplabv3": deeplabv3p(volume, 2),
#     "hrnet": hrnet(volume, 2),
# }


def create_model(input, num_class=2):
    """构建训练模型.

    Parameters
    ----------
    input : paddle.data
        输入的 placeholder.
    num_class : int
        输出分类有几类.

    Returns
    -------
    type
        构建好的模型.

    """
    if cfg.TRAIN.ARCHITECTURE == "unet_plain":
        return unet_plain(input, num_class, cfg.TRAIN.INPUT_SIZE)
    if cfg.TRAIN.ARCHITECTURE == "res_unet":
        return res_unet(input, num_class, cfg.TRAIN.INPUT_SIZE)
    if cfg.TRAIN.ARCHITECTURE == "hrnet":
        return hrnet(input, num_class)
    if cfg.TRAIN.ARCHITECTURE == "deeplabv3p":
        return deeplabv3p(input, num_class)
    raise Exception("错误的网络类型： {}".format(cfg.TRAIN.ARCHITECTURE))
