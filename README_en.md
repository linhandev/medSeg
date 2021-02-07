# MedSeg
English | [简体中文](./README_cn.md)

Medical image segmentation toolkit based on PaddlePaddle framework. Our target is to implement various 2D and 3D model architectures, various loss function and data augmentation methods. This is still a work in progress but has achieved promising results on liver segmentation and aorta segmentation. The development plans can be seen in the [Project](https://github.com/davidlinhl/medSeg/projects/1)

## Project Structure
Currently this project contains only 2D segmentation models. The structure is as follows.

- medseg: Promary code
  - train.py: Training pipeline
  - aug.py: Data augmentation
  - loss.py: Various model loss
  - eval.py: Various metrics to evaluate segmentation result
  - vis.py: Visualize results
  - models: Currently only 2D models
- tool: Useful scripts
  - train: Tools for converting scan format, generating 2D slices from scan etc.
  - infer: Tools used after inference, merging segmentation results from slices, voting model fusion etc.
- config: Training configurations
All configurations can be found in [utils/config.py](https://github.com/davidlinhl/medSeg/blob/master/medseg/utils/config.py)

## Usage
### Environment Set Up
Install project dependencies with
```shell
pip install -r requirements.txt
```
Instructions for installing PaddlePaddle-GPU can be found on PaddlePaddle's [official home page](https://www.paddlepaddle.org.cn/)

### Preprocess
Preprocess 3D scans either into 2D slices or 3D patches. Applying WWWC or other slice-wise augmentation can also be done here.
```shell
python medseg/prep_3d.py -c config/lits.yaml
```

### Training
The training script contains several choices. Run with -h command to see details about them. If u r training with CPU only, don't include the --use_gpu command.
```shell
python medseg/train.py -c config/lits.yaml --use_gpu --do_eval
```

### Inference
The last step is doing inference with previously trained model. The script would perform inference on all data under specified path and perform inference. Currently supports nii format only.
```shell
python medseg/infer.py -c config/lits.yaml --use_gpu
```

### Evaluation and Else
After getting inference results, you may want to know how well the model performs. We have an evaluation script with multiple metrics implemented with medpy.
```shell
python medseg/eval.py -c config/eval.yaml
```
For aorta, specifically, we also have scripts for measuring blood vessel diameter and reporting aorta aneurysm.
```shell
python tool/infer/2d_diameter.py
python tool/infer/aorta.py
```
These two scripts combined can calculate aorta diameter and report aorta aneurysm based on it.

We have an [Aistudio](https://aistudio.baidu.com/aistudio/projectdetail/250994) project that have all the data and environment ready.

Should u have any question in using this toolkit, u can contact the developer at linhandev@qq.com
