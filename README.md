# medSeg
中文 | [English](./README_en.md)

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
- eval.py 对分割结果进行评估，支持基本所有医学影像常用2d/3d metric

#### tool 工具脚本
tool中提供了一些实用的工具脚本，[train](./train)目录下主要用于训练前的预处理，[infer](./infer)目录下的主要用于推理和后处理。
- dcm2nii.py 将dcm格式转换为nii格式
- folder_split.py 将两个文件夹的训练文件夹随机分成训练/测试/验证集
- mhd2nii.py 将mhd格式数据转换成nii格式
- nii2png.py 将nii格式数据转换成png
- to_512.py 将所有数据插值到512×512大小
- flood_fill.py 用漫水法填满分割标签
- to_pinyin.py 将所有文件名由中文转拼音
- zip_dataset.py 将一个文件夹压缩，压缩包不超过指定大小
- 2d_diameter.py 在2d平面内，以平行线夹的方式测量血管管径
- merge.py 将多种前景的分割结果合并进一个文件
- vote.py 用投票法对结果进行合并

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
