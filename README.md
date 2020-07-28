# medSeg
仿照百度[PaddleSeg](https://github.com/paddlepaddle/paddleseg)结构实现的一个医学影像方向分割任务开发套件。主要目标是实现多种2D和3D网络，多种loss和数据增强策略。目前项目还在开发中，但是已经能在肝脏分割场景下做到 .94 的IOU。2.5D P-Unet项目基于这个套件实现。开发计划见[Project](https://github.com/davidlinhl/medSeg/projects/1)

## 项目结构
#### medseg 项目主体

- prep_3d.py prep_2d.py 分别对3D和2D的输入数据进行预处理，形成npy格式的训练数据
- loss.py 定义loss
- models 定义模型
- aug.py 定义数据增强方法
- train.py 训练网络
- infer.py 用训练完的模型进行推理
- vis.py 对数据进行可视化
- eval.py 对分割结果进行评估

#### tool 工具脚本

- merge.py 将针对同一个数据的多个前景分割结果进行融合
- thresh_search.py 利用验证集搜索最合适的结果划分阈值
- vote.py 利用投票的方式进行多个分割结果的融合
- zip_dataset.py 对数据集进行分包压缩，满足每个包不超过一定大小，解压后可以还原目录结构

#### config 配置文件
所有配置参考[config.py](https://github.com/davidlinhl/medSeg/blob/master/medseg/utils/config.py)

## 使用方法
### 配置环境
安装环境依赖:
```shell
pip install -r requirements.txt
```
paddle框架的安装参考[paddle官网](https://www.paddlepaddle.org.cn/)
如果进行训练需要有数据，目前项目主要面向lits调试，在aistudio上可以找到。[训练集](https://aistudio.baidu.com/aistudio/datasetDetail/10273) [测试集](https://aistudio.baidu.com/aistudio/datasetDetail/10292)

数据集下载，解压之后将所有的训练集volume放到一个文件夹，所有的训练集label放到一个文件夹，测试集volume放到一个文件夹。修改 lits.yaml 中对应的路径。

### 预处理
配置完毕需要首先进行数据预处理，这里主要是将数据统一成npz格式，方便后续训练。也可以在这一步结合一些预处理步骤对3D CT数据可以做窗口化，3D旋转。
```shell
python medseg/prep_3d.py -c config/lits.yaml
```
### 训练
网络用预处理后的数据进行训练，训练提供一些参数，-h 可以显示。如果用的是cpu版本的paddle，不要添加 --use_gpu 参数。
```shell
python medseg/train.py -c config/lits.yaml --use_gpu --do_eval
```
### 预测
最后一步是用训练好的网络进行预测，要配置好模型权重的路径，按照上一步实际输出的路径进行修改。代码会读取inference路径下所有的nii逐个进行预测。目前支持的数据格式有 .nii, .nii.gz。
```shell
python infer.py -c config/lits.yaml --use_gpu
```
<br>

这个项目在aistudio中有完整的环境，fork项目可以直接运行，[项目地址](https://aistudio.baidu.com/aistudio/projectdetail/250994)中运行

# 更新日志
* 2020.6.21
**v0.2.0**
* 项目整体修改为使用配置文件，丰富功能，增加可视化

* 2020.5.1
**v0.1.0**
* 在lits数据集上跑通预处理，训练，预测脚本

如项目使用中有任何问题，欢迎加入 Aistudio医学兴趣组，在群中提问，也可以和更多大佬一起学习和进步。

<div align="center">
  <img src="https://i.loli.net/2020/05/28/HFwS4eNxJPAp72Y.jpg" alt="2132453929.jpg" style="zoom:40%;" />
</div>
