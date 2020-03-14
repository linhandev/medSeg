from __future__ import print_function

import sys
import os
import argparse

import math
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.utils.plot import Ploter

from models.unet_simple import unet_simple
from models.deeplabv3p import deeplabv3p
from models.unet_base import unet_base
from util import *
import random
from paddle.fluid.layers import log
from config import *

def parse_args():
	parser = argparse.ArgumentParser("train")
	parser.add_argument('--use_gpu',type=bool,default=True,help="是否用GPU")
	parser.add_argument('--num_epochs', type=int, default=20, help="多少个epoch")
	parser.add_argument('--net', type=str, default="unet_base", help="选择使用网络类型")

	args = parser.parse_args()
	return args


def data_reader(part_start=0,part_end=8):
	data_names=listdir(preprocess_path)
	data_part=data_names[len(data_names)*part_start//10:len(data_names)*part_end//10]
	random.shuffle(data_part)

	def reader():
		for data_name in data_part:
			data=np.load(preprocess_path+data_name)
			vol=data[0:3,:,:]
			lab=data[3,:,:]
			yield (vol.reshape(3,512,512).astype("float32") ,lab.reshape(1,512,512).astype("int32"))
	return reader


def mean_iou(pred, label, num_classes=2):
	'''
		返回miou
	'''

	pred = fluid.layers.argmax(pred, axis=1)
	pred = fluid.layers.cast(pred, 'int32')
	label = fluid.layers.cast(label, 'int32')
	miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
	return miou


def weighed_binary_cross_entropy(y,y_predict,beta=2,epsilon=1e-6):
	'''
		返回 wce loss
		beta标记的是希望positive类给到多少的权重，如果positive少，beta给大于1相当与比0的类更重视
	'''
	y=fluid.layers.clip(y,epsilon,1-epsilon)
	y_predict=fluid.layers.clip(y_predict,epsilon,1-epsilon)

	ylogp=elementwise_mul(y,log(y_predict))
	betas=fill_constant(ylogp.shape,"float32",beta)
	ylogp=elementwise_mul(betas,ylogp)

	ones=fill_constant(y_predict.shape,"float32",1)
	ylogp=elementwise_add(ylogp,elementwise_mul(elementwise_sub(ones,y) , log(elementwise_sub(ones,y_predict) ) ) )

	zeros=fill_constant(y_predict.shape,"float32",0)
	return elementwise_sub(zeros,ylogp)

def focal_loss(y_predict,y,alpha=0.75,gamma=2,epsilon=1e-6):
	'''
		alpha 变大，对前景类惩罚变大，更加重视
		gamma 变大，对信心大的例子更加忽略，学习难的例子
	'''
	y=fluid.layers.clip(y,epsilon,1-epsilon)
	y_predict=fluid.layers.clip(y_predict,epsilon,1-epsilon)

	return -1 * (alpha * pow( (1 - y_predict) ,gamma ) * y * log(y_predict) + (1 - alpha) * pow( y_predict , gamma ) * ( 1 - y ) * log(1 - y_predict))


def create_loss(predict, label, num_classes=2):
	predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
	predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
	predict = fluid.layers.softmax(predict)
	label = fluid.layers.reshape(label, shape=[-1, 1])
	label = fluid.layers.cast(label,"int64")

	dice_loss = fluid.layers.dice_loss(predict, label)

	label = fluid.layers.cast(label,"float32")

	# ce_loss = fluid.layers.cross_entropy(predict, label)
	focal = focal_loss(predict,label)

	return fluid.layers.reduce_mean(focal + dice_loss)


def main():
	batch_size = 16
	num_epochs = args.num_epochs
	net_type=args.net

	# 训练ckpt和inf模型路径
	param_base_dir=os.path.join(code_base_dir,'params')
	param_base_dir=os.path.join(param_base_dir,net_type)
	infer_param_path=os.path.join(param_base_dir,"inf")
	ckpt_param_path=os.path.join(param_base_dir,"ckpt")
	print(infer_param_path)
	print(ckpt_param_path)

	train_reader=paddle.batch(paddle.reader.shuffle(data_reader(),int(batch_size*1.5)),batch_size)
	test_reader=paddle.batch(paddle.reader.shuffle(data_reader(8,9),int(batch_size*1.5)),batch_size)

	train_program=fluid.Program()
	train_init=fluid.Program()

	with fluid.program_guard(train_program,train_init):
		image = fluid.layers.data(name='image', shape=[3,512,512], dtype='float32')
		label = fluid.layers.data(name='label', shape=[1,512,512], dtype='int32')
		train_loader=fluid.io.DataLoader.from_generator(feed_list=[image,label],capacity=batch_size)
		test_loader=fluid.io.DataLoader.from_generator(feed_list=[image,label],capacity=batch_size)

		if net_type == "unet_simple":
			prediction = unet_simple(image,2,[512,512])
		elif net_type == "unet_base":
			prediction = unet_base(image,2,[512,512])
		elif net_type == "deeplabv3":
			prediction = deeplabv3p(image,2)
		else:
			print("错误的网络类型")
			sys.exit(0)

		avg_loss = create_loss(prediction,label,2)

		miou = mean_iou(prediction, label, 2)

		decay=paddle.fluid.regularizer.L2Decay(0.1)
		# optimizer = fluid.optimizer.SGD(learning_rate=0.0005,regularization=decay)
		# optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.02,regularization=decay)
		# optimizer = fluid.optimizer.RMSProp(learning_rate=0.1,momentum=0.8,centered=True, regularization=decay)

		optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.003)


		# optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.006, momentum=0.8,regularization=decay)

		optimizer.minimize(avg_loss)

	place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
	exe = fluid.Executor(place)
	exe.run(train_init)
	# fluid.io.load_persistables(exe, ckpt_param_path, train_init)

	exe_test=fluid.Executor(place)

	test_program = train_program.clone(for_test=True)


	# train_program=fluid.CompiledProgram(train_program).with_data_parallel(loss_name=avg_loss.name)
	test_program=fluid.CompiledProgram(test_program).with_data_parallel(loss_name=avg_loss.name)

	train_loader.set_sample_list_generator(train_reader, places=place)
	test_loader.set_sample_list_generator(test_reader, places=place)

	feeder=fluid.DataFeeder(place=place,feed_list=[image,label])

	step = 1
	best_miou=0

	train_prompt="Train_miou"
	test_prompt="Test_miou"

	plot_prompt = Ploter(train_prompt, test_prompt)

	for pass_id in range(num_epochs):
			for data_train in train_loader():
				avg_loss_value,miou_value = exe.run(train_program,feed=data_train,fetch_list=[avg_loss,miou])

				if step % 10 == 0:
					print("\t\tTrain pass %d, Step %d, Cost %f, Miou %f" % (pass_id, step, avg_loss_value[0],miou_value[0]))

				# if step % 10 ==0:
					# plot_prompt.append(train_prompt, step, miou_value[0])
					# plot_prompt.plot()

				eval_miou=0
				if step % 300 == 0:
					auc_metric = fluid.metrics.Auc("AUC")
					test_losses=[]
					test_mious=[]
					for _,test_data in enumerate(test_loader()):
						# print(test_data)
						# input("pause")

						_, test_loss, test_miou = exe_test.run(test_program, feed=test_data, fetch_list=[prediction, avg_loss, miou])
						test_losses.append(test_loss[0])
						test_mious.append(test_miou[0])

					eval_miou=np.average(np.array(test_mious))
					# plot_prompt.append(test_prompt, step, eval_miou)
					# plot_prompt.plot()

					print("Test loss: %f ,miou: %f" % (np.average(np.array(test_losses)),eval_miou ) )

				if math.isnan(float(avg_loss_value[0])):
					sys.exit("got NaN loss, training failed.")

				if step%200 == 0 and param_base_dir is not None and eval_miou>best_miou:
					best_miou=eval_miou
					print("Saving params of step: %d" % step)
					fluid.io.save_inference_model(infer_param_path, feeded_var_names=['image'], target_vars=[prediction], executor=exe,main_program=train_program)
					fluid.io.save_persistables(exe,ckpt_param_path,train_program)
				step += 1
	print(best_miou)

if __name__ == '__main__':
	args = parse_args()
	main()
