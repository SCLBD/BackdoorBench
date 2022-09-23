'''
This file implements the defense method called finetuning (clp), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
	1. basic setting: args
	2. attack result(model, train data, test data)
	3. clp defense:
		
	4. test the result and get ASR, ACC, RC 
'''

import argparse
import logging
import os
import random
import sys




sys.path.append('../')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

#from utils import args
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
#from utils.input_aware_utils import progress_bar
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

def get_args():
	#set the basic parameter
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--device', type=str, help='cuda, cpu')
	parser.add_argument('--checkpoint_load', type=str)
	parser.add_argument('--checkpoint_save', type=str)
	parser.add_argument('--log', type=str)
	parser.add_argument("--data_root", type=str)

	parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
	parser.add_argument("--num_classes", type=int)
	parser.add_argument("--input_height", type=int)
	parser.add_argument("--input_width", type=int)
	parser.add_argument("--input_channel", type=int)

	parser.add_argument('--epochs', type=int)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument("--num_workers", type=float)
	parser.add_argument('--lr', type=float)
	parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

	parser.add_argument('--poison_rate', type=float)
	parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
	parser.add_argument('--target_label', type=int)

	parser.add_argument('--model', type=str, help='resnet18')
	parser.add_argument('--random_seed', type=int, help='random seed')
	parser.add_argument('--index', type=str, help='index of clean data')
	parser.add_argument('--result_file', type=str, help='the location of result')

	parser.add_argument('--yaml_path', type=str, default="./config/defense/clp/config.yaml", help='the path of yaml')

	#set the parameter for the clp defense
	parser.add_argument('--u', type=float, help='the default value of u')
	parser.add_argument('--u_step', type=float, help='the step of u')

	arg = parser.parse_args()

	print(arg)
	return arg

def CLP_prune(net, u):
	for m in net.modules():
		if isinstance(m, nn.BatchNorm2d):
			std = m.running_var.sqrt()
			weight = m.weight

			channel_lips = []
			for idx in range(weight.shape[0]):
				w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
				channel_lips.append(torch.svd(w)[1].max())
			channel_lips = torch.Tensor(channel_lips)

			index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

			params = m.state_dict()
			for idx in index:
				params['weight'][idx] = 0
			m.load_state_dict(params)
		
		# Convolutional layer should be followed by a BN layer by default
		elif isinstance(m, nn.Conv2d):
			conv = m
	return net

def test_result(arg,testloader_cl,testloader_bd,model,criterion):
	model.eval()
	
	total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
	for i, (inputs, labels) in enumerate(testloader_cl):
		inputs, labels = inputs.to(arg.device), labels.to(arg.device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		test_loss += loss.item()

		total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
		total_clean_test += inputs.shape[0]
		avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
		#progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
	print('| Test Acc: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
	logging.info('| Test Acc: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
			
	total_backdoor_test, total_backdoor_correct_test, test_loss = 0, 0, 0
	for i, (inputs, labels) in enumerate(testloader_bd):
		inputs, labels = inputs.to(arg.device), labels.to(arg.device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		test_loss += loss.item()

		total_backdoor_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
		total_backdoor_test += inputs.shape[0]
		avg_acc_clean = float(total_backdoor_correct_test.item() * 100.0 / total_backdoor_test)
		#progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
	print('| Test Asr: {:.3f}%({}/{})'.format(avg_acc_clean, total_backdoor_correct_test, total_backdoor_test))
	logging.info('| Test Asr: {:.3f}%({}/{})'.format(avg_acc_clean, total_backdoor_correct_test, total_backdoor_test))
	return list([total_clean_test, total_clean_correct_test,total_backdoor_test, total_backdoor_correct_test])

def clp(args,result,config):
	logFormatter = logging.Formatter(
		fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%Y-%m-%d:%H:%M:%S',
	)
	logger = logging.getLogger()
	# logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
	if args.log is not None and args.log != '':
		fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
	else:
		fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
	fileHandler.setFormatter(logFormatter)
	logger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	logger.addHandler(consoleHandler)

	logger.setLevel(logging.INFO)
	logging.info(pformat(args.__dict__))

	fix_random(args.random_seed)

	# Prepare model, optimizer, scheduler
	model = generate_cls_model(args.model,args.num_classes)
	model.load_state_dict(result['model'])
	model.to(args.device)
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	# if args.lr_scheduler == 'ReduceLROnPlateau':
	#     scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
	# elif args.lr_scheduler ==  'CosineAnnealingLR':
	#     scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100) 
	criterion = nn.CrossEntropyLoss()
	
	tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	x = result['bd_test']['x']
	y = result['bd_test']['y']
	data_bd_test = list(zip(x,y))
	data_bd_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_bd_test,
		poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)
	data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

	tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	x = result['clean_test']['x']
	y = result['clean_test']['y']
	data_clean_test = list(zip(x,y))
	data_clean_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_clean_test,
		poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)
	data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
	te_result = test_result(args,data_clean_loader,data_bd_loader,model,criterion)
	logging.info('Origin model ACC: {} ASR: {}'.format(te_result[1]/te_result[0],te_result[3]/te_result[2]))
	u_set = np.linspace(0,10,args.u_step)
	for i in u_set:
		model_new = copy.deepcopy(model)
		model_new.eval()
		model_new = CLP_prune(model_new,i)
		te_result = test_result(args,data_clean_loader,data_bd_loader,model_new,criterion)
		logging.info('Origin model ACC: {} ASR: {}'.format(te_result[1]/te_result[0],te_result[3]/te_result[2]))
	if args.u is not None:
		CLP_prune(model,args.u)

	te_result = test_result(args,data_clean_loader,data_bd_loader,model,criterion)
	logging.info('Origin model ACC: {} ASR: {}'.format(te_result[1]/te_result[0],te_result[3]/te_result[2]))
	result = {}
	result['model'] = model_new
	return result


if __name__ == '__main__':
	### 1. basic setting: args
	args = get_args()
	with open(args.yaml_path, 'r') as stream: 
		config = yaml.safe_load(stream) 
	config.update({k:v for k,v in args.__dict__.items() if v is not None})
	args.__dict__ = config
	args.num_classes = get_num_classes(args.dataset)
	args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
	args.img_size = (args.input_height, args.input_width, args.input_channel)
	
	save_path = '/record/' + args.result_file
	if args.checkpoint_save is None:
		args.checkpoint_save = save_path + '/record/defence/clp/'
		if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
			os.makedirs(os.getcwd() + args.checkpoint_save) 
	if args.log is None:
		args.log = save_path + '/saved/clp/'
		if not (os.path.exists(os.getcwd() + args.log)):
			os.makedirs(os.getcwd() + args.log)  
	args.save_path = save_path

	### 2. attack result(model, train data, test data)
	result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
	
	print("Continue training...")
	### 3. clp defense:
	result_defense = clp(args,result,config)

	### 4. test the result and get ASR, ACC, RC 
	result_defense['model'].eval()
	result_defense['model'].to(args.device)
	tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	x = result['bd_test']['x']
	y = result['bd_test']['y']
	data_bd_test = list(zip(x,y))
	data_bd_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_bd_test,
		poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)
	data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

	asr_acc = 0
	for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
		inputs, labels = inputs.to(args.device), labels.to(args.device)
		outputs = result_defense['model'](inputs)
		pre_label = torch.max(outputs,dim=1)[1]
		asr_acc += torch.sum(pre_label == labels)
	asr_acc = asr_acc/len(data_bd_test)

	tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	x = result['clean_test']['x']
	y = result['clean_test']['y']
	data_clean_test = list(zip(x,y))
	data_clean_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_clean_test,
		poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)
	data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

	clean_acc = 0
	for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
		inputs, labels = inputs.to(args.device), labels.to(args.device)
		outputs = result_defense['model'](inputs)
		pre_label = torch.max(outputs,dim=1)[1]
		clean_acc += torch.sum(pre_label == labels)
	clean_acc = clean_acc/len(data_clean_test)

	tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	x = result['bd_test']['x']
	robust_acc = -1
	if 'original_targets' in result['bd_test']:
		y_ori = result['bd_test']['original_targets']
		if y_ori is not None:
			if len(y_ori) != len(x):
				y_idx = result['bd_test']['original_index']
				y = y_ori[y_idx]
			else :
				y = y_ori
			data_bd_test = list(zip(x,y))
			data_bd_testset = prepro_cls_DatasetBD(
				full_dataset_without_transform=data_bd_test,
				poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
				bd_image_pre_transform=None,
				bd_label_pre_transform=None,
				ori_image_transform_in_loading=tran,
				ori_label_transform_in_loading=None,
				add_details_in_preprocess=False,
			)
			data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
		
			robust_acc = 0
			for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
				inputs, labels = inputs.to(args.device), labels.to(args.device)
				outputs = result_defense['model'](inputs)
				pre_label = torch.max(outputs,dim=1)[1]
				robust_acc += torch.sum(pre_label == labels)
			robust_acc = robust_acc/len(data_bd_test)

	if not (os.path.exists(os.getcwd() + f'{save_path}/clp/')):
		os.makedirs(os.getcwd() + f'{save_path}/clp/')
	torch.save(
	{
		'model_name':args.model,
		'model': result_defense['model'].cpu().state_dict(),
		'asr': asr_acc,
		'acc': clean_acc,
		'ra': robust_acc
	},
	os.getcwd() + f'{save_path}/clp/defense_result.pt'
	)
