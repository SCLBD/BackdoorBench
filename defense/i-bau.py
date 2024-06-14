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
Adversarial unlearning of backdoors via implicit hypergradient
This file is modified based on the following source:
link : https://github.com/YiZeng623/I-BAU/

@inproceedings{zeng2021adversarial,
  title={Adversarial Unlearning of Backdoors via Implicit Hypergradient},
  author={Zeng, Yi and Chen, Si and Park, Won and Mao, Zhuoqing and Jin, Ming and Jia, Ruoxi},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

The defense method is called i-bau.
The license is bellow the code

The update include:
	1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. use clean samples from training (align other defense Settings)
basic sturcture for defense method:
	1. basic setting: args
	2. attack result(model, train data, test data)
	3. i-bau defense:
		a. get some clean data
		b. unlearn the backdoor model by the pertubation
	4. test the result and get ASR, ACC, RC 
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO:怎么查看包的相对路径和绝对路径
sys.path.append('../')
sys.path.append(os.getcwd())

# TODO:修改yaml文件

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch, given_dataloader_test
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
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

class i_bau(defense):
	r"""Adversarial unlearning of backdoors via implicit hypergradient
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor model
    4. i-bau defense:
        a. train the adversarial purturbaion by the clean data using the hypergradient
		b. unlearn the backdoor model by the pertubation
		c. repeat a and b for several rounds
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        i_bau.add_arguments(parser)
        args = parser.parse_args()
        i_bau_method = i_bau(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = i_bau_method.defense(args.result_file)
    
    .. Note::
        @article{zeng2021adversarial,
		title={Adversarial unlearning of backdoors via implicit hypergradient},
		author={Zeng, Yi and Chen, Si and Park, Won and Mao, Z Morley and Jin, Ming and Jia, Ruoxi},
		journal={arXiv preprint arXiv:2110.03735},
		year={2021}
		}

    Args:
        baisc args: in the base class
        ratio (float): the ratio of clean data loader
		index (str): index of clean data
		optim (str): type of outer loop optimizer utilized (default: Adam) to train the adversarial purturbaion
		n_rounds (int): the maximum number of unlearning rounds and the number of fixed point iterations (default: 10)
		K (int): the maximum number of fixed point iterations (default: 10)

    
    """ 

	def __init__(self,args):
		with open(args.yaml_path, 'r') as f:
			defaults = yaml.safe_load(f)

		defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

		args.__dict__ = defaults

		args.terminal_info = sys.argv

		args.num_classes = get_num_classes(args.dataset)
		args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
		args.img_size = (args.input_height, args.input_width, args.input_channel)
		args.dataset_path = f"{args.dataset_path}/{args.dataset}"

		# TODO:直接用self.args好不好用
		self.args = args

		if 'result_file' in args.__dict__ :
			if args.result_file is not None:
				self.set_result(args.result_file)

	def add_arguments(parser):
		parser.add_argument('--device', type=str, help='cuda, cpu')
		parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
		parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
		parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
		parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

		parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
		parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
		parser.add_argument('--log', type=str, help='the location of log')
		parser.add_argument("--dataset_path", type=str, help='the location of data')
		parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
		parser.add_argument('--result_file', type=str, help='the location of result')
	
		parser.add_argument('--epochs', type=int)
		parser.add_argument('--batch_size', type=int)
		parser.add_argument("--num_workers", type=float)
		parser.add_argument('--lr', type=float)
		parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
		parser.add_argument('--steplr_stepsize', type=int)
		parser.add_argument('--steplr_gamma', type=float)
		parser.add_argument('--steplr_milestones', type=list)
		parser.add_argument('--model', type=str, help='resnet18')
		
		parser.add_argument('--client_optimizer', type=int)
		parser.add_argument('--sgd_momentum', type=float)
		parser.add_argument('--wd', type=float, help='weight decay of sgd')
		parser.add_argument('--frequency_save', type=int,
						help=' frequency_save, 0 is never')

		parser.add_argument('--random_seed', type=int, help='random seed')
		parser.add_argument('--yaml_path', type=str, default="./config/defense/i-bau/config.yaml", help='the path of yaml')

		#set the parameter for the i-bau defense
		parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
		parser.add_argument('--index', type=str, help='index of clean data')
		parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
		parser.add_argument('--n_rounds', type=int, help='the maximum number of unelarning rounds')
		parser.add_argument('--K', type=int, help='the maximum number of fixed point iterations')

		

		
	def set_result(self, result_file):
		attack_file = 'record/' + result_file
		save_path = 'record/' + result_file + '/defense/i-bau/'
		if not (os.path.exists(save_path)):
			os.makedirs(save_path)
		# assert(os.path.exists(save_path))    
		self.args.save_path = save_path
		if self.args.checkpoint_save is None:
			self.args.checkpoint_save = save_path + 'checkpoint/'
			if not (os.path.exists(self.args.checkpoint_save)):
				os.makedirs(self.args.checkpoint_save) 
		if self.args.log is None:
			self.args.log = save_path + 'log/'
			if not (os.path.exists(self.args.log)):
				os.makedirs(self.args.log)  
		self.result = load_attack_result(attack_file + '/attack_result.pt')
	def set_trainer(self, model):
		self.trainer = PureCleanModelTrainer(
			model,
		)

	def set_logger(self):
		args = self.args
		logFormatter = logging.Formatter(
			fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
			datefmt='%Y-%m-%d:%H:%M:%S',
		)
		logger = logging.getLogger()

		fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
		fileHandler.setFormatter(logFormatter)
		logger.addHandler(fileHandler)

		consoleHandler = logging.StreamHandler()
		consoleHandler.setFormatter(logFormatter)
		logger.addHandler(consoleHandler)

		logger.setLevel(logging.INFO)
		logging.info(pformat(args.__dict__))

		try:
			logging.info(pformat(get_git_info()))
		except:
			logging.info('Getting git info fails.')
	
	def set_devices(self):
		# self.device = torch.device(
		# 	(
		# 		f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
		# 		# since DataParallel only allow .to("cuda")
		# 	) if torch.cuda.is_available() else "cpu"
		# )
		self.device= self.args.device
	def mitigation(self):
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		model = generate_cls_model(self.args.model,self.args.num_classes)
		model.load_state_dict(self.result['model'])
		if "," in self.device:
			model = torch.nn.DataParallel(
				model,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			self.args.device = f'cuda:{model.device_ids[0]}'
			model.to(self.args.device)
		else:
			model.to(self.args.device)
		### TODO: opt
		# outer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
		# criterion = nn.CrossEntropyLoss()
		outer_opt, scheduler = argparser_opt_scheduler(model, self.args)
		# criterion = nn.CrossEntropyLoss()
		self.set_trainer(model)
		# criterion = argparser_criterion(args)

		
		# a. get some clean data
		logging.info("We use clean train data, the original paper use clean test data.")
		transforms_list = []
		transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
		transforms_list.append(transforms.ToTensor())
		transforms_list.append(get_dataset_normalization(args.dataset))
		tran = transforms.Compose(transforms_list)
		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
		data_all_length = len(clean_dataset)
		ran_idx = choose_index(self.args, data_all_length) 
		log_index = self.args.log + 'index.txt'
		np.savetxt(log_index, ran_idx, fmt='%d')
		clean_dataset.subset(ran_idx)
		data_set_without_tran = clean_dataset
		data_set_o = self.result['clean_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		# data_set_o = prepro_cls_DatasetBD_v2(
		#     full_dataset_without_transform=data_set,
		#     poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
		#     bd_image_pre_transform=None,
		#     bd_label_pre_transform=None,
		#     ori_image_transform_in_loading=train_tran,
		#     ori_label_transform_in_loading=None,
		#     add_details_in_preprocess=False,
		# )
		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
		trainloader = data_loader
		
		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

		# self.trainer.train_with_test_each_epoch(
		#     train_data = trainloader,
		#     test_data = data_clean_loader,
		#     adv_test_data = data_bd_loader,
		#     end_epoch_num = self.args.epochs,
		#     criterion = criterion,
		#     optimizer = optimizer,
		#     scheduler = scheduler,
		#     device = self.args.device,
		#     frequency_save = self.args.frequency_save,
		#     save_folder_path = self.args.checkpoint_save,
		#     save_prefix = 'defense',
		#     continue_training_path = None,
		# )
		# test_result(args,data_clean_loader,data_bd_loader,0,model,criterion)

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
		for index, (images, labels, original_index, poison_indicator, original_targets) in enumerate(trainloader):
			images_list.append(images)
			labels_list.append(labels)
		inner_opt = GradientDescent(loss_inner, 0.1)

		train_loss_list = []
		train_mix_acc_list = []
		train_clean_acc_list = []
		train_asr_list = []
		train_ra_list = []

		clean_test_loss_list = []
		bd_test_loss_list = []
		ra_test_loss_list = []
		test_acc_list = []
		test_asr_list = []
		test_ra_list = []

		# b. unlearn the backdoor model by the pertubation
		logging.info("=> Conducting Defence..")
		model.eval()
		agg = Metric_Aggregator()
		for round in range(args.n_rounds):
			# batch_pert = torch.zeros_like(data_clean_testset[0][:1], requires_grad=True, device=args.device)
			batch_pert = torch.zeros([1,args.input_channel,args.input_height,args.input_width], requires_grad=True, device=args.device)
			batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
		
			for images, labels, original_index, poison_indicator, original_targets in trainloader:
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
			
			clean_test_loss_avg_over_batch, \
			bd_test_loss_avg_over_batch, \
			ra_test_loss_avg_over_batch, \
			test_acc, \
			test_asr, \
			test_ra = self.eval_step(
				model,
				data_clean_loader,
				data_bd_loader,
				args,
			)

			agg({
				"epoch": round,

				"clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
				"bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
				"ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
				"test_acc": test_acc,
				"test_asr": test_asr,
				"test_ra": test_ra,
			})


			clean_test_loss_list.append(clean_test_loss_avg_over_batch)
			bd_test_loss_list.append(bd_test_loss_avg_over_batch)
			ra_test_loss_list.append(ra_test_loss_avg_over_batch)
			test_acc_list.append(test_acc)
			test_asr_list.append(test_asr)
			test_ra_list.append(test_ra)

			general_plot_for_epoch(
				{
					"Test C-Acc": test_acc_list,
					"Test ASR": test_asr_list,
					"Test RA": test_ra_list,
				},
				save_path=f"{args.save_path}i-bau_acc_like_metric_plots.png",
				ylabel="percentage",
			)

			general_plot_for_epoch(
				{
					"Test Clean Loss": clean_test_loss_list,
					"Test Backdoor Loss": bd_test_loss_list,
					"Test RA Loss": ra_test_loss_list,
				},
				save_path=f"{args.save_path}i-bau_loss_metric_plots.png",
				ylabel="percentage",
			)

			agg.to_dataframe().to_csv(f"{args.save_path}i-bau_df.csv")
		agg.summary().to_csv(f"{args.save_path}i-bau_df_summary.csv")
		# self.trainer.train_with_test_each_epoch_on_mix(
		#     trainloader,
		#     data_clean_loader,
		#     data_bd_loader,
		#     args.epochs,
		#     criterion=criterion,
		#     optimizer=optimizer,
		#     scheduler=scheduler,
		#     device=self.device,
		#     frequency_save=args.frequency_save,
		#     save_folder_path=args.save_path,
		#     save_prefix='i-bau',
		#     amp=args.amp,
		#     prefetch=args.prefetch,
		#     prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
		#     non_blocking=args.non_blocking,
		# )
		
		result = {}
		result['model'] = model
		save_defense_result(
			model_name=args.model,
			num_classes=args.num_classes,
			model=model.cpu().state_dict(),
			save_path=args.save_path,
		)
		return result

	def eval_step(
			self,
			netC,
			clean_test_dataloader,
			bd_test_dataloader,
			args,
	):
		clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
			netC,
			clean_test_dataloader,
			criterion=torch.nn.CrossEntropyLoss(),
			non_blocking=args.non_blocking,
			device=self.device,
			verbose=0,
		)
		clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
		test_acc = clean_metrics['test_acc']
		bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
			netC,
			bd_test_dataloader,
			criterion=torch.nn.CrossEntropyLoss(),
			non_blocking=args.non_blocking,
			device=self.args.device,
			verbose=0,
		)
		bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
		test_asr = bd_metrics['test_acc']

		bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
		ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
			netC,
			bd_test_dataloader,
			criterion=torch.nn.CrossEntropyLoss(),
			non_blocking=args.non_blocking,
			device=self.args.device,
			verbose=0,
		)
		ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
		test_ra = ra_metrics['test_acc']
		bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

		return clean_test_loss_avg_over_batch, \
				bd_test_loss_avg_over_batch, \
				ra_test_loss_avg_over_batch, \
				test_acc, \
				test_asr, \
				test_ra

	def defense(self,result_file):
		self.set_result(result_file)
		self.set_logger()
		result = self.mitigation()
		return result
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=sys.argv[0])
	i_bau.add_arguments(parser)
	args = parser.parse_args()
	i_bau_method = i_bau(args)
	if "result_file" not in args.__dict__:
		args.result_file = 'defense_test_badnet'
	elif args.result_file is None:
		args.result_file = 'defense_test_badnet'
	result = i_bau_method.defense(args.result_file)