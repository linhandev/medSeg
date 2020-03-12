# liverSeg
基于paddle框架的肝脏及肝脏肿瘤分割。
目前这个项目还是一个work in progress但是已经能在LiTS肝脏任务上达到 .95 的准确率，觉得效果还不错。
项目结构
- config.py包含项目配置，主要是文件路径
- preprocess.py进行数据预处理，将3D体数据保存为2.5Dnpy
- train.py进行Unet训练
- infer.py进行前向推理


这个项目在aistudio中有完整的环境，也可以直接到[aistudio](https://aistudio.baidu.com/aistudio/projectdetail/250994)中运行
