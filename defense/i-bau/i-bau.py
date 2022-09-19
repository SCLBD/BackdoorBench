# MIT License

# Copyright (c) 2021 Yi Zeng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''
This file is modified based on the following source:
link : https://github.com/YiZeng623/I-BAU/
The defense method is called i-bau.
The license is bellow the code

basic sturcture for defense method:
	1. basic setting: args
	2. attack result(model, train data, test data)
	3. i-bau defense:
		a. get some clean data
		b. unlearn the backdoor model by the pertubation
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
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
#from utils.input_aware_utils import progress_bar
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

import torchvision.transforms as transforms
import random
from itertools import repeat

from typing import List, Callable
from torch import Tensor
from torch.autograd import grad as torch_grad

'''
Based on the paper 'On the Iteration Complexity of Hypergradient Computation,' this code was created.
Source: https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py
Original Author: Riccardo Grazzi
'''


class DifferentiableOptimizer:
	def __init__(self, loss_f, dim_mult, data_or_iter=None):
		"""
		Args:
			loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
			data_or_iter: (x, y) or iterator over the data needed for loss_f
		"""
		self.data_iterator = None
		if data_or_iter:
			self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

		self.loss_f = loss_f
		self.dim_mult = dim_mult
		self.curr_loss = None

	def get_opt_params(self, params):
		opt_params = [p for p in params]
		opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
		return opt_params

	def step(self, params, hparams, create_graph):
		raise NotImplementedError

	def __call__(self, params, hparams, create_graph=True):
		with torch.enable_grad():
			return self.step(params, hparams, create_graph)

	def get_loss(self, params, hparams):
		if self.data_iterator:
			data = next(self.data_iterator)
			self.curr_loss = self.loss_f(params, hparams, data)
		else:
			self.curr_loss = self.loss_f(params, hparams)
		return self.curr_loss

class GradientDescent(DifferentiableOptimizer):
	def __init__(self, loss_f, step_size, data_or_iter=None):
		super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
		self.step_size_f = step_size if callable(step_size) else lambda x: step_size

	def step(self, params, hparams, create_graph):
		loss = self.get_loss(params, hparams)
		sz = self.step_size_f(hparams)
		return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
	grads = torch.autograd.grad(loss, params, create_graph=create_graph)
	return [w - step_size * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
	grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
								retain_graph=retain_graph, create_graph=create_graph)

	def grad_or_zeros(grad, var):
		return torch.zeros_like(var) if grad is None else grad

	return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
	grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
	grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

	return grad_outer_w, grad_outer_hparams

def update_tensor_grads(hparams, grads):
	for l, g in zip(hparams, grads):
		if l.grad is None:
			l.grad = torch.zeros_like(l)
		if g is not None:
			l.grad += g


def fixed_point(params: List[Tensor],
				hparams: List[Tensor],
				K: int ,
				fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
				outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
				tol=1e-10,
				set_grad=True,
				stochastic=False) -> List[Tensor]:
	"""
	Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).
	Args:
		params: the output of the inner solver procedure.
		hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
		K: the maximum number of fixed point iterations
		fp_map: the fixed point map which defines the inner problem
		outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
		tol: end the method earlier when  the normed difference between two iterates is less than tol
		set_grad: if True set t.grad to the hypergradient for every t in hparams
		stochastic: set this to True when fp_map is not a deterministic function of its inputs
	Returns:
		the list of hypergradients for each element in hparams
	"""

	params = [w.detach().requires_grad_(True) for w in params]
	o_loss = outer_loss(params, hparams)
	grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

	if not stochastic:
		w_mapped = fp_map(params, hparams)

	vs = [torch.zeros_like(w) for w in params]
	vs_vec = cat_list_to_tensor(vs)
	for k in range(K):
		vs_prev_vec = vs_vec

		if stochastic:
			w_mapped = fp_map(params, hparams)
			vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
		else:
			vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

		vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
		vs_vec = cat_list_to_tensor(vs)
		if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
			break

	if stochastic:
		w_mapped = fp_map(params, hparams)

	grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
	grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

	if set_grad:
		update_tensor_grads(hparams, grads)

	return grads

def cat_list_to_tensor(list_tx):
	return torch.cat([xx.reshape([-1]) for xx in list_tx])

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
	
	parser.add_argument('--lr',type=float)
	parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

	parser.add_argument('--poison_rate', type=float)
	parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
	parser.add_argument('--target_label', type=int)

	parser.add_argument('--model', type=str, help='resnet18')
	parser.add_argument('--random_seed', type=int, help='random seed')
	parser.add_argument('--index', type=str, help='index of clean data')
	parser.add_argument('--result_file', type=str, help='the location of result')

	parser.add_argument('--yaml_path', type=str, default="./config/defense/i-bau/config.yaml", help='the path of yaml')

	#set the parameter for the i-bau defense
	parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
	## hyper params
	### TODO config optimizer 改框架之后放到前面统一起来
	parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
	parser.add_argument('--n_rounds', type=int, help='the maximum number of unelarning rounds')
	parser.add_argument('--K', type=int, help='the maximum number of fixed point iterations')
	# parser.add_argument('--dataset', default='cifar10', help='the dataset to use')
	# parser.add_argument('--poi_path', default='./checkpoint/badnets_8_02_ckpt.pth', help='path of the poison model need to be unlearn')
	# parser.add_argument('--log_path', default='./unlearn_logs', help='path of the log file')
	# parser.add_argument('--device', type=str, default='4,5,6,7', help='Device to use. Like cuda, cuda:0 or cpu')
	# parser.add_argument('--batch_size', type=int, default=100, help='batch size of unlearn loader')
	# parser.add_argument('--unl_set', default=None, help='extra unlearn dataset, if None then use test data')
	
	arg = parser.parse_args()

	print(arg)
	return arg
	
def i_bau(args,result,config):
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

	# a. get some clean data
	logging.info("We use clean train data, the original paper use clean test data.")
	transforms_list = []
	transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
	transforms_list.append(transforms.ToTensor())
	transforms_list.append(get_dataset_normalization(args.dataset))
	tran = transforms.Compose(transforms_list)
	x = result['clean_train']['x']
	y = result['clean_train']['y']
	data_all_length = len(y)
	ran_idx = choose_index(args, data_all_length) 
	log_index = os.getcwd() + args.log + 'index.txt'
	np.savetxt(log_index, ran_idx, fmt='%d')
	data_set = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
	data_set_o = prepro_cls_DatasetBD(
		full_dataset_without_transform=data_set,
		poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
		bd_image_pre_transform=None,
		bd_label_pre_transform=None,
		ori_image_transform_in_loading=tran,
		ori_label_transform_in_loading=None,
		add_details_in_preprocess=False,
	)
	
	data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
	trainloader = data_loader

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
	data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)

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
	data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)

	# Prepare model, optimizer, scheduler
	model = generate_cls_model(args.model,args.num_classes)
	model.load_state_dict(result['model'])
	model.to(args.device)

	### TODO: adam and sgd
	outer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()
	
	test_result(args,data_clean_loader,data_bd_loader,0,model,criterion)

	### define the inner loss L2
	def loss_inner(perturb, model_params):
		### TODO: cpu training and multiprocessing
		images = images_list[0].to(args.device)
		labels = labels_list[0].long().to(args.device)
		#per_img = torch.clamp(images+perturb[0],min=0,max=1)
		per_img = images+perturb[0]
		per_logits = model.forward(per_img)
		loss = F.cross_entropy(per_logits, labels, reduction='none')
		loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
		return loss_regu

	### define the outer loss L1
	def loss_outer(perturb, model_params):
		### TODO: cpu training and multiprocessing
		portion = 0.01
		images, labels = images_list[batchnum].to(args.device), labels_list[batchnum].long().to(args.device)
		patching = torch.zeros_like(images, device='cuda')
		number = images.shape[0]
		rand_idx = random.sample(list(np.arange(number)),int(number*portion))
		patching[rand_idx] = perturb[0]
		#unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
		unlearn_imgs = images+patching
		logits = model(unlearn_imgs)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(logits, labels)
		return loss

	images_list, labels_list = [], []
	for index, (images, labels) in enumerate(trainloader):
		images_list.append(images)
		labels_list.append(labels)
	inner_opt = GradientDescent(loss_inner, 0.1)


	# b. unlearn the backdoor model by the pertubation
	logging.info("=> Conducting Defence..")
	model.eval()
	for round in range(args.n_rounds):
		# batch_pert = torch.zeros_like(data_clean_testset[0][:1], requires_grad=True, device=args.device)
		batch_pert = torch.zeros([1,args.input_channel,args.input_height,args.input_width], requires_grad=True, device=args.device)
		batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
	
		for images, labels in trainloader:
			images = images.to(args.device)
			ori_lab = torch.argmax(model.forward(images),axis = 1).long()
			# per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
			per_logits = model.forward(images+batch_pert)
			loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
			loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(batch_pert),2)
			batch_opt.zero_grad()
			loss_regu.backward(retain_graph = True)
			batch_opt.step()

		#l2-ball
		# pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
		pert = batch_pert

		#unlearn step         
		for batchnum in range(len(images_list)): 
			outer_opt.zero_grad()
			fixed_point(pert, list(model.parameters()), args.K, inner_opt, loss_outer) 
			outer_opt.step()
		test_result(args,data_clean_loader,data_bd_loader,round,model,criterion)
		
	result = {}
	result['model'] = model
	return result





def test_result(arg,testloader_cl,testloader_bd,epoch,model,criterion):
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
	print('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))
	logging.info('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))
			
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
	print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_backdoor_correct_test, total_backdoor_test))
	logging.info('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_backdoor_correct_test, total_backdoor_test))

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
		args.checkpoint_save = save_path + '/record/defence/i-bau/'
		if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
			os.makedirs(os.getcwd() + args.checkpoint_save) 
	if args.log is None:
		args.log = save_path + '/saved/i-bau/'
		if not (os.path.exists(os.getcwd() + args.log)):
			os.makedirs(os.getcwd() + args.log)  
	args.save_path = save_path

	### 2. attack result(model, train data, test data)
	result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
	
	print("Continue training...")
	### 3. i-bau defense:
	result_defense = i_bau(args,result,config)

	### 4. test the result and get ASR, ACC, RC 
	result_defense['model'].eval()
	result_defense['model'].to(args.device)
	### I-BAU use norm 
	transforms_list = []
	transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
	transforms_list.append(transforms.ToTensor())
	transforms_list.append(get_dataset_normalization(args.dataset))
	tran = transforms.Compose(transforms_list)
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

	transforms_list = []
	transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
	transforms_list.append(transforms.ToTensor())
	transforms_list.append(get_dataset_normalization(args.dataset))
	tran = transforms.Compose(transforms_list)
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

	transforms_list = []
	transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
	transforms_list.append(transforms.ToTensor())
	transforms_list.append(get_dataset_normalization(args.dataset))
	tran = transforms.Compose(transforms_list)
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

	if not (os.path.exists(os.getcwd() + f'{save_path}/i-bau/')):
		os.makedirs(os.getcwd() + f'{save_path}/i-bau/')
	torch.save(
	{
		'model_name':args.model,
		'model': result_defense['model'].cpu().state_dict(),
		'asr': asr_acc,
		'acc': clean_acc,
		'ra': robust_acc
	},
	os.getcwd() + f'{save_path}/i-bau/defense_result.pt'
	)
