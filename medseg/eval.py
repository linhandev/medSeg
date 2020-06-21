# 在验证集上对模型的多种指标进行评估
from medpy import metric


metric.dc(pred, label)
metric.binary.jc(pred, label)
metric.ravd(label, pred)
