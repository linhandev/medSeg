# liverSeg
基于paddle框架的肝脏及肝脏肿瘤分割。这个项目目前还是work in progress，但是在肝脏上已经做到.92的准确率. 目前支持2.5D Unet,Res-Unet和DeepLab. 后期会加入更多网络结构和loss做成一个小框架。

## 项目结构
- config.py 包含项目配置，主要是文件路径
- preprocess.py 进行数据预处理，主要将3D体数据保存为2.5D npy
- train.py 进行Unet训练
- infer.py 进行前向推理

## 使用方法
### 配置环境
安装环境依赖:
```shell
pip install -r requirements.txt
```
paddle框架的安装参考[paddle官网](https://www.paddlepaddle.org.cn/)
如果进行训练需要有lits数据集，在aistudio上可以找到。[训练集](https://aistudio.baidu.com/aistudio/datasetDetail/10273) [测试集](https://aistudio.baidu.com/aistudio/datasetDetail/10292) <br>
数据集下载，解压之后将所有的训练集volume放到一个文件夹，所有的训练集label放到一个文件夹，测试集volume放到一个文件夹。修改 config.py 中对应的路径。

### 预处理
配置完毕需要首先进行数据预处理，这里主要是将数据从3D转为2.5D的slice，方便后续训练，也可以在这一步结合一些预处理步骤，比如窗口化，3D旋转之类的。这里前面的路径都配置好了的话应该不会有问题。
```shell
python preprocess.py
```
### 训练
网络用预处理后的数据进行训练，训练提供一些参数，可以查看代码或者 -h 显示。如果用的是cpu版本的paddle，use_gpu设成 False。
```shell
python train.py --use_gpu = True --num_epochs = 20
```
### 预测
最后一步是用训练好的网络进行预测，模型权重的路径在 infer.py 中注意修改，按照上一步实际输出的路径进行修改。代码会读取inference路径下所有的nii逐个进行预测。目前支持的数据格式有 .nii, .nii.gz。预训练权重在[这个](https://github.com/davidlinhl/weights)项目中可以下载。
```shell
python infer.py -use_gpu = True
```
<br>

这个项目在aistudio中有完整的环境，fork项目可以直接运行，[项目地址](https://aistudio.baidu.com/aistudio/projectdetail/250994)中运行
