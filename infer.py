from __future__ import print_function

import sys
import argparse

import numpy
import cv2

import paddle
from paddle import fluid 
from util import * 
import nibabel as nib
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="number of epochs.")
    args = parser.parse_args()
    return args

data_names=listdir(preprocess_path)

# def data_reader(part_start=9,part_end=10):
#     data_part=data_names[len(data_names)*part_start//10:len(data_names)*part_end//10]
    
#     def reader():
#         for data_name in data_part:
#             data=np.load(preprocess_path+data_name)
#             vol=data[0:3,:,:]
        
#             yield vol.reshape(3,512,512).astype("float32") 
#     return reader

def main():
    use_cuda = args.use_gpu
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    infer_param_path="/home/aistudio/work/params/unet_base_16_iou965/inf/"
    
    # infer_reader=data_reader()
    # infer_reader=paddle.batch(infer_reader,1)
    
    # infer
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(infer_param_path, infer_exe)
        
        inf_volumes = os.listdir(inference_path)
        for inf_volume in inf_volumes:
            inf_path = os.path.join(inference_path, inf_volume)
            print(inf_path)

            volf = nib.load(inf_path)
            volume = np.array(volf.get_fdata())
            volume = volume.clip(-1024, 1024)
            inference = np.zeros(volume.shape)
            # print(volume.shape)
            # print(inference.shape)

            for i in tqdm(range(1,volume.shape[2]-1) ):
                data = volume[:,:,i-1:i+2]
                data=data.swapaxes(0,2).reshape([1,3,512,512]).astype('float32')

                assert feed_target_names[0] == 'image'
                result = infer_exe.run(inference_program,feed={feed_target_names[0]: data },fetch_list=fetch_targets)
                result = np.array(result)
                result = result.reshape([2,512,512])
                result = result.swapaxes(0,2)[:,:,1]
                result[result > 0.9] = 1
                result[result < 0.9] = 0
                result = result* 255
                inference[:,:,i] = result
                cv2.imwrite("result.png",result)
                
            inference_file = nib.Nifti1Image(inference, volf.affine)
            print(os.path.join(inference_label_path, inf_volume))
            nib.save(inference_file, os.path.join(inference_label_path, inf_volume))
            input("pause")

if __name__ == '__main__':
    args = parse_args()
    main()


