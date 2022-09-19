'''
This file is modified based on the following source:
link : https://github.com/IBM/model-sanitization.
The defense method is called MCR.

The update include:
	1. data preprocess and dataset setting
	2. model setting
	3. args and config
	4. save process
	5. new standard: robust accuracy
basic sturcture for defense method:
	1. basic setting: args
	2. attack result(model, train data, test data)
	3. mcr defense:
		TODO : update the processing and 'update'
	4. test the result and get ASR, ACC, RC 
	
	This data-split method is different from the benchmark, so it only support the cifar10. In the future, it will be updated
'''

import logging
import random
import time

from calendar import c
from unittest.mock import sentinel
from torchvision import transforms

import torch
import logging
import argparse
import sys
import os

import tqdm


sys.path.append('../')
sys.path.append(os.getcwd())
from defense.mcr.curve_models import curves
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result

sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat
from tqdm import tqdm
import numpy as np
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from defense.mcr import curve_models

### TODO: set two models or fine tune model
### TODO: checkpoint for mcr model
def get_curve_class(args):
	### TODO: preactresnet18 and vgg19 need to be support 
	if args.model == 'preactresnet18':
		net = getattr(curve_models, 'PreResNet110')
	elif args.model == 'vgg19':
		net = getattr(curve_models, 'VGG19BN')
	# elif args.model_name == 'densenet161':
	# 	net = getattr(curve_models, args.model)
	# elif args.model_name == 'mobilenet_v3_large':
	# 	net = getattr(curve_models, args.model)
	# elif args.model_name == 'efficientnet_b3':
	# 	net = getattr(curve_models, args.model)
	else:
		raise SystemError('NO valid model match in function generate_cls_model!')

	return net

def get_mcr_dataset(args, result, shuffle_train=True):
	### TODO: use clean_train data to train the curve model
	x = result['clean_test']['x']
	y = result['clean_test']['y']
	data_all_length = round(len(y)/2)
	ran_idx = choose_index(args, data_all_length) 
	log_index = os.getcwd() + args.log + 'index.txt'
	np.savetxt(log_index,ran_idx, fmt='%d')

	aa = np.array(range(len(x)))
	bb = np.random.shuffle(aa)

	x_train_part = [x[ii] for ii in aa[0:data_all_length][ran_idx]] 
	y_train_part = [y[ii] for ii in aa[0:data_all_length][ran_idx]] 

	i1 = data_all_length // len(ran_idx)
	i2 = data_all_length % len(ran_idx)

	x_train_half = []
	y_train_half = []

	for ii in range(i1):
		x_train_half += x_train_part
		y_train_half += y_train_part

	if i2 != 0:
		x_train_half += [x_train_part[i] for i in range(data_all_length - i1 * len(ran_idx))]
		y_train_half += [y_train_part[i] for i in range(data_all_length - i1 * len(ran_idx))]

	x_train = x_train_half + x_train_half
	y_train = y_train_half + y_train_half

	x_test = [x[ii] for ii in aa[data_all_length:2*data_all_length]] + [x[ii] for ii in aa[data_all_length:2*data_all_length]]
	y_test = [y[ii] for ii in aa[data_all_length:2*data_all_length]] + [y[ii] for ii in aa[data_all_length:2*data_all_length]]

	train_ori = list(zip(x_train, y_train))
	test_ori = list(zip(x_test, y_test))

	tran_train = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
	data_trainset = prepro_cls_DatasetBD(
		full_dataset_without_transform=train_ori,
		poison_idx=np.zeros(len(train_ori)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran_train,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)

	tran_test = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
	data_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=test_ori,
		poison_idx=np.zeros(len(test_ori)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran_test,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)

	x = result['bd_test']['x']
	y = result['bd_test']['y']
	data_bd_test = list(zip(x,y))
	data_bd_testset = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_bd_test,
		poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran_test,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)


	return {
			   'train': torch.utils.data.DataLoader(
				   data_trainset,
				   batch_size=args.batch_size,
				   shuffle=shuffle_train,
				   num_workers=args.num_workers,
				   pin_memory=True
			   ),
			   'test': torch.utils.data.DataLoader(
				   data_testset,
				   batch_size=args.batch_size,
				   shuffle=False,
				   num_workers=args.num_workers,
				   pin_memory=True
			   ),
			   'test_bd': torch.utils.data.DataLoader(
				   data_bd_testset,
				   batch_size=args.batch_size,
				   shuffle=False,
				   num_workers=args.num_workers,
				   pin_memory=True
			   ),
			   'testset': data_testset,
		   }

def train(args, train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
	loss_sum = 0.0
	correct = 0.0

	num_iters = len(train_loader)
	model.train()
	for iter, (input, target) in enumerate(train_loader):
		if lr_schedule is not None:
			lr = lr_schedule(iter / num_iters)
			adjust_learning_rate(optimizer, lr)
		input = input.to(args.device)
		target = target.to(args.device)

		output = model(input)
		loss = criterion(output, target)
		if regularizer is not None:
			loss += regularizer(model)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_sum += loss.item() * input.size(0)
		pred = output.data.argmax(1, keepdim=True)
		correct += pred.eq(target.data.view_as(pred)).sum().item()
	aa = len(train_loader.dataset)
	bb = loss_sum
	return {
		'loss': loss_sum / len(train_loader.dataset),
		'accuracy': correct * 100.0 / len(train_loader.dataset),
	}

def test(args, test_loader, model, criterion, regularizer=None, **kwargs):
	loss_sum = 0.0
	nll_sum = 0.0
	correct = 0.0

	model.eval()

	for input, target in test_loader:
		input = input.to(args.device)
		target = target.to(args.device)

		output = model(input, **kwargs)
		nll = criterion(output, target)
		loss = nll.clone()
		if regularizer is not None:
			loss += regularizer(model)

		nll_sum += nll.item() * input.size(0)
		loss_sum += loss.item() * input.size(0)
		pred = output.data.argmax(1, keepdim=True)
		correct += pred.eq(target.data.view_as(pred)).sum().item()

	return {
		'nll': nll_sum / len(test_loader.dataset),
		'loss': loss_sum / len(test_loader.dataset),
		'accuracy': correct * 100.0 / len(test_loader.dataset),
	}


def test_epoch(args, testloader, model, criterion, epoch, word):
	'''test the student model with regard to test data for each epoch
	args:
		Contains default parameters
	testloader:
		the dataloader of clean test data or backdoor test data
	model:
		the student model
	criterion:
		criterion during the train process
	epoch:
		current epoch
	word:
		'bd' or 'clean'
	'''
	model.eval()

	total_clean, total_clean_correct, test_loss = 0, 0, 0

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(testloader):
			inputs, labels = inputs.to(args.device), labels.to(args.device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			test_loss += loss.item()

			total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
			total_clean += inputs.shape[0]
			avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
	   
	if word == 'bd':
		logging.info(f'Test {word} ASR: {avg_acc_clean} ({total_clean_correct}/{total_clean})')
	if word == 'clean':
		logging.info(f'Test {word} ACC: {avg_acc_clean} ({total_clean_correct}/{total_clean})')

	return test_loss / (i + 1), avg_acc_clean


def adjust_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def get_args():
	# set the basic parameter
	### TODO: basic parameter setting
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

	parser.add_argument('--attack', type=str)
	parser.add_argument('--poison_rate', type=float)
	parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
	parser.add_argument('--target_label', type=int)
	parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

	parser.add_argument('--model', type=str, help='resnet18')
	parser.add_argument('--random_seed', type=int, help='random seed')
	### TODO : split the data by our benchmark
	# parser.add_argument('--index', type=str, help='index of clean data')
	parser.add_argument('--result_file', type=str, help='the location of result')

	parser.add_argument('--yaml_path', type=str, default="./config/defense/mcr/config.yaml", help='the path of yaml')

	#set the parameter for the mcr defense
	parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
	parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
					help='curve type to use (default: None)')
	parser.add_argument('--num_bends', type=int, default=3, metavar='N',
						help='number of curve bends (default: 3)')
	parser.add_argument('--init_start', type=str, default='Res_single_true_10_same1/checkpoint-100.pt', metavar='CKPT',
						help='checkpoint to init start point (default: None)')
	parser.add_argument('--fix_start', dest='fix_start', action='store_true', default=True,
						help='fix start point (default: off)')
	parser.add_argument('--init_end', type=str, default='Res_single_true_10_same2/checkpoint-100.pt', metavar='CKPT',
						help='checkpoint to init end point (default: None)')
	parser.add_argument('--fix_end', dest='fix_end', action='store_true', default=True,
						help='fix end point (default: off)')
	parser.set_defaults(init_linear=True)
	parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
						help='turns off linear initialization of intermediate points (default: on)')
	
	parser.add_argument('--ft_epochs', type=int)
	parser.add_argument('--ft_lr',type=float)
	parser.add_argument('--ft_lr_scheduler', type=str, help='the scheduler of lr for fine tuning')
	
	parser.add_argument('--momentum', type=float)
	parser.add_argument('--wd', type=float, help='weight decay of sgd')

	arg = parser.parse_args()

	print(arg)
	return arg



def mcr(args, result, config):
	### set logger
	logFormatter = logging.Formatter(
		fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%Y-%m-%d:%H:%M:%S',
	)
	logger = logging.getLogger()
	if args.log is not None and args.log != '':
		fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
	else:
		fileHandler = logging.FileHandler('./log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
	fileHandler.setFormatter(logFormatter)
	logger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	logger.addHandler(consoleHandler)

	logger.setLevel(logging.INFO)
	logging.info(pformat(args.__dict__))

	fix_random(args.random_seed)

	# os.makedirs(args.dir, exist_ok=True)
	# with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
	#     f.write(' '.join(sys.argv))
	#     f.write('\n')

	# torch.backends.cudnn.benchmark = True
	# torch.manual_seed(args.seed)
	# torch.cuda.manual_seed(args.seed)

	loaders = get_mcr_dataset(args, result)

	architecture = get_curve_class(args)

	# if args.curve is None:
	#     model = architecture.base(num_classes=arg.num_classes, **architecture.kwargs)
	# else:
	assert(args.curve is not None)
	curve = getattr(curves, args.curve)
	model = curves.CurveNet(
		args.num_classes,
		curve,
		architecture.curve,
		args.num_bends,
		args.fix_start,
		args.fix_end,
		architecture_kwargs=architecture.kwargs,
	)
	base_model = None
	
	# for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
	#     if path is not None:
	#         if base_model is None:
				
	#         checkpoint = torch.load(path)
	#         print('Loading %s as point #%d' % (path, k))
	#         base_model.load_state_dict(checkpoint['model_state'])
	#         model.import_base_parameters(base_model, k)
	base_model = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
	k = 0
	base_model.load_state_dict(result['model'])
	model.import_base_parameters(base_model, k)

	ft_model = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
	k = args.num_bends - 1
	ft_model.load_state_dict(result['model'])
	start_epoch = 0
	optimizer_ft = torch.optim.SGD(ft_model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
	if args.ft_lr_scheduler == 'ReduceLROnPlateau':
		scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft)
	elif args.ft_lr_scheduler ==  'CosineAnnealingLR':
		scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft, T_max=100)
	criterion_ft = nn.CrossEntropyLoss()
	for epoch in tqdm(range(start_epoch, args.ft_epochs)):
		ft_model.to(args.device)
		batch_loss = []
		for i, (inputs,labels) in enumerate(trainloader):  # type: ignore
			ft_model.train()
			ft_model.to(args.device)
			inputs, labels = inputs.to(args.device), labels.to(args.device)
			outputs = ft_model(inputs)
			loss = criterion_ft(outputs, labels)
			batch_loss.append(loss.item())
			optimizer_ft.zero_grad()
			loss.backward()
			optimizer_ft.step()
			del loss, inputs, outputs
			torch.cuda.empty_cache()
		one_epoch_loss = sum(batch_loss)/len(batch_loss)
		if args.lr_scheduler == 'ReduceLROnPlateau':
			scheduler_ft.step(one_epoch_loss)
		elif args.lr_scheduler ==  'CosineAnnealingLR':
			scheduler_ft.step()

		# evaluate on testing set
		test_loss, test_acc_cl = test_epoch(args, loaders['test'], ft_model, criterion_ft, epoch, 'clean')
		test_loss, test_acc_bd = test_epoch(args, loaders['test_bd'], ft_model, criterion_ft, epoch, 'bd')
		# remember best precision and save checkpoint
		logging.info(f'Teacher_Epoch{epoch}: clean_acc:{test_acc_cl} asr:{test_acc_bd}')
	model.import_base_parameters(ft_model, k)

	k = 0
	if args.init_linear:
		print('Linear initialization.')
		model.init_linear()
	model.to(args.device)


	def learning_rate_schedule(base_lr, epoch, total_epochs):
		alpha = epoch / total_epochs
		if alpha <= 0.5:
			factor = 1.0
		elif alpha <= 0.9:
			factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
		else:
			factor = 0.01
		return factor * base_lr


	criterion = F.cross_entropy
	regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

	### TODO:adam or sgd
	optimizer = torch.optim.SGD(
		filter(lambda param: param.requires_grad, model.parameters()),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=args.wd if args.curve is None else 0.0
	)

	start_epoch = 1
	

	# columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']


	# has_bn = utils.check_bn(model)
	test_res = {'loss': None, 'accuracy': None, 'nll': None}
	test_bd_res = {'loss': None, 'accuracy': None, 'nll': None}

	logging.info('Epoch \t lr \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
	best_acc = 0
	for epoch in range(start_epoch, args.epochs + 1):
		# time_ep = time.time()
		model.to(args.device)

		lr = learning_rate_schedule(args.lr, epoch, args.epochs)
		# lr = args.lr
		adjust_learning_rate(optimizer, lr)

		train_res = train(args, loaders['train'], model, optimizer, criterion, regularizer)
		t = torch.FloatTensor([0.0]).cuda()
		t.data.fill_(0.3)
		test_res = test(args, loaders['test'], model, criterion, regularizer, t=t)
		test_bd_res = test(args, loaders['test_bd'], model, criterion, regularizer, t=t)

		# if epoch % args.save_freq == 0:
		#     utils.save_checkpoint(
		#         args.dir,
		#         epoch,
		#         model_state=model.state_dict(),
		#         optimizer_state=optimizer.state_dict()
		#     )

		# time_ep = time.time() - time_ep
		# values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
		#         test_res['accuracy'], time_ep]

		logging.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
			epoch, lr, train_res['loss'], train_res['accuracy'],test_bd_res['nll'], test_bd_res['accuracy'], test_res['nll'],
				test_res['accuracy']))

		# table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
		# if epoch % 40 == 1 or epoch == start_epoch:
		#     table = table.split('\n')
		#     table = '\n'.join([table[1]] + table)
		# else:
		#     table = table.split('\n')[2]
		# print(table)

		if best_acc < test_res['accuracy']:
			best_acc = test_res['accuracy']
			best_asr = test_bd_res['accuracy']
			torch.save(
			{
				'model_name':args.model,
				'model': model.cpu().state_dict(),
				'asr': best_asr,
				'acc': best_acc,
				'curve': args.curve,
				'optimizer_state': optimizer.state_dict()
			},
			f'.{args.checkpoint_save}defense_result.pt'
			)

	# if args.epochs % args.save_freq != 0:
	#     utils.save_checkpoint(
	#         args.dir,
	#         args.epochs,
	#         model_state=model.state_dict(),
	#         optimizer_state=optimizer.state_dict()
	#     )
	### TODO: cross validation to choose t
	t = torch.FloatTensor([0.0]).cuda()
	t.data.fill_(0.3)
	### TODO: weight debug
	# weights = model.weights(t)
	# previous_weights = weights.copy()

	# update_bn(loaders['train'], model, t=t)
	result = {}
	result['model'] = model
	result['t'] = t
	return result

def isbatchnorm(module):
	return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
		   issubclass(module.__class__, curves._BatchNorm)

def _check_bn(module, flag):
	if isbatchnorm(module):
		flag[0] = True

def check_bn(model):
	flag = [False]
	model.apply(lambda module: _check_bn(module, flag))
	return flag[0]


def reset_bn(module):
	if isbatchnorm(module):
		module.reset_running_stats()

def reset_bn(module):
	if isbatchnorm(module):
		module.reset_running_stats()


def _get_momenta(module, momenta):
	if isbatchnorm(module):
		momenta[module] = module.momentum


def _set_momenta(module, momenta):
	if isbatchnorm(module):
		module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
	### TODO: cuda and cpu
	device = 'cuda'
	if not check_bn(model):
		return
	model.train()
	momenta = {}
	model.apply(reset_bn)
	model.apply(lambda module: _get_momenta(module, momenta))
	num_samples = 0
	for input, _ in loader:
		input = input.to(device)
		batch_size = input.data.size(0)

		momentum = batch_size / (num_samples + batch_size)
		for module in momenta.keys():
			module.momentum = momentum

		model(input, **kwargs)
		num_samples += batch_size

	model.apply(lambda module: _set_momenta(module, momenta))


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
		args.checkpoint_save = save_path + '/record/defence/mcr/'
		if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
			os.makedirs(os.getcwd() + args.checkpoint_save) 
	if args.log is None:
		args.log = save_path + '/saved/mcr/'
		if not (os.path.exists(os.getcwd() + args.log)):
			os.makedirs(os.getcwd() + args.log)  
	args.save_path = save_path

	### 2. attack result(model, train data, test data)
	result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
	
	print("Continue training...")
	### 3. mcr defense:
	result_defense = mcr(args,result,config)

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
		outputs = result_defense['model'](inputs,result_defense['t'])
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
		outputs = result_defense['model'](inputs,result_defense['t'])
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
				outputs = result_defense['model'](inputs,result_defense['t'])
				pre_label = torch.max(outputs,dim=1)[1]
				robust_acc += torch.sum(pre_label == labels)
			robust_acc = robust_acc/len(data_bd_test)

	if not (os.path.exists(os.getcwd() + f'{save_path}/mcr/')):
		os.makedirs(os.getcwd() + f'{save_path}/mcr/')
	torch.save(
	{
		'model_name':args.model,
		'model': result_defense['model'].cpu().state_dict(),
		'asr': asr_acc,
		'acc': clean_acc,
		'ra': robust_acc,
		't':result_defense['t'].cpu(),
	},
	os.getcwd() + f'{save_path}/mcr/defense_result.pt'
	)