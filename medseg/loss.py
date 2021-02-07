import paddle
import paddle.fluid as fluid
from paddle.fluid.layers import log


def mean_iou(pred, label, num_classes=2):
    """
		计算miou
	"""
    pred = fluid.layers.argmax(pred, axis=1)
    pred = fluid.layers.cast(pred, "int32")
    label = fluid.layers.cast(label, "int32")
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
    return miou


def weighed_binary_cross_entropy(y, y_predict, beta=2, epsilon=1e-6):
    """
		返回 wce loss
		beta标记的是希望positive类给到多少的权重，如果positive少，beta给大于1相当与比0的类更重视
	"""
    y = fluid.layers.clip(y, epsilon, 1 - epsilon)
    y_predict = fluid.layers.clip(y_predict, epsilon, 1 - epsilon)

    ylogp = fluid.layers.elementwise_mul(y, log(y_predict))
    betas = fluid.layers.fill_constant(ylogp.shape, "float32", beta)
    ylogp = fluid.layers.elementwise_mul(betas, ylogp)

    ones = fluid.layers.fill_constant(y_predict.shape, "float32", 1)
    ylogp = fluid.layers.elementwise_add(
        ylogp, elementwise_mul(elementwise_sub(ones, y), log(elementwise_sub(ones, y_predict)))
    )

    zeros = fluid.layers.fill_constant(y_predict.shape, "float32", 0)
    return fluid.layers.elementwise_sub(zeros, ylogp)


def focal_loss(y_predict, y, alpha=0.85, gamma=2, epsilon=1e-6):
    """
		alpha 变大，对前景类惩罚变大，更加重视
		gamma 变大，对信心大的例子更加忽略，学习难的例子
	"""
    y = fluid.layers.clip(y, epsilon, 1 - epsilon)
    y_predict = fluid.layers.clip(y_predict, epsilon, 1 - epsilon)

    return -1 * (
        alpha * fluid.layers.pow((1 - y_predict), gamma) * y * log(y_predict)
        + (1 - alpha) * fluid.layers.pow(y_predict, gamma) * (1 - y) * log(1 - y_predict)
    )


def create_loss(predict, label, num_classes=2):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    predict = fluid.layers.softmax(predict)
    label = fluid.layers.reshape(label, shape=[-1, 1])
    label = fluid.layers.cast(label, "int64")
    dice_loss = fluid.layers.dice_loss(predict, label)

    # label = fluid.layers.cast(label, "int64")

    ce_loss = fluid.layers.cross_entropy(predict, label)
    # focal = focal_loss(predict, label)

    return fluid.layers.reduce_mean(ce_loss + dice_loss)
