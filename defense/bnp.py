'''
Pre-activation Distributions Expose Backdoor Neurons

This file is modified based on the following source:
link : https://github.com/RJ-T/NIPS2022_EP_BNP.
The defense method is called bnp.

@article{zheng2022pre,
            title={Pre-activation Distributions Expose Backdoor Neurons},
            author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
            journal={Advances in Neural Information Processing Systems},
            volume={35},
            pages={18667--18680},
            year={2022}}

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. reconstruct the layer norm for convnext and transformer
    7. draw the corresponding images of asr and acc according to different proportions
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. dde defense:
        a. calculate the entropy of each norm layer
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import copy
import os,sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
import utils.defense_utils.mbns.mbns_model as mbns_model
from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

def batch_entropy(x, step_size=0.1):
	n_bars = int((x.max()-x.min())/step_size)
	entropy = 0
	for n in range(n_bars):
		num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
		p = num / x.shape[-1]
		entropy += - p * p.log().nan_to_num(0)
	return entropy

def bnp_defense(net, u, trainloader, args):
	clean_data_loader = trainloader['clean_train']
	bd_data_loader = trainloader['bd_train']
	net.eval()
	bd_data = iter(bd_data_loader).next()[0].to(args.device)
	mixture_data = iter(clean_data_loader).next()[0].to(args.device)
	params = net.state_dict()
	for m in net.modules():
		if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
			m.collect_stats = True
		if isinstance(m, nn.LayerNorm):
			m.collect_stats_clean = True
			m.collect_stats_bd = False
	with torch.no_grad():
		net(mixture_data)
	for m in net.modules():
		if isinstance(m, nn.LayerNorm):
			m.collect_stats_bd = True
			m.collect_stats_clean = False
	with torch.no_grad():
		net(bd_data)
		for name, m in net.named_modules():
			if isinstance(m, nn.BatchNorm2d):
				var_2 = m.running_var
				var_1 = m.batch_var
				mean_2 = m.running_mean
				mean_1 = m.batch_mean
				kl_div = (var_2/var_1).log() + (var_1+(mean_1-mean_2).pow(2))/(2*var_2) - 1/2
				index = (kl_div>kl_div.mean() + u*kl_div.std())

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0
			elif isinstance(m, nn.LayerNorm):
				# We use layer norm to subsitute batch norm in convnext_model and vit_model
				var_2 = m.batch_var_bd
				var_1 = m.batch_var_clean
				mean_2 = m.batch_mean_bd
				mean_1 = m.batch_mean_clean
				kl_div = (var_2/var_1).log() + (var_1+(mean_1-mean_2).pow(2))/(2*var_2) - 1/2
				index = (kl_div>kl_div.mean() + u*kl_div.std())

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0

	net.load_state_dict(params)

def get_mbns_network(
	model_name: str,
	num_classes: int = 10,
	**kwargs,
):
	if model_name == 'preactresnet18':
		net = mbns_model.preact_mbns.PreActResNet18(num_classes = num_classes, **kwargs)
	elif model_name == 'vgg19_bn':
		net = mbns_model.vgg_mbns.vgg19_bn(num_classes = num_classes,  **kwargs)
	elif model_name == 'densenet161':
		net = mbns_model.den_mbns.densenet161(num_classes= num_classes, **kwargs)
	elif model_name == 'mobilenet_v3_large':
		net = mbns_model.mobilenet_mbns.mobilenet_v3_large(num_classes= num_classes, **kwargs)
	elif model_name == 'efficientnet_b3':
		net = mbns_model.eff_mbns.efficientnet_b3(num_classes= num_classes, **kwargs)
	elif model_name == 'convnext_tiny':
		try :
			net = mbns_model.conv_mbns.convnext_tiny(num_classes= num_classes, 
			)
		except :
			net = mbns_model.conv_new_mbns.convnext_tiny(num_classes= num_classes, 
			)
	elif model_name == 'vit_b_16':
		try :
			from torchvision.transforms import Resize
			net = mbns_model.vit_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
		except :
			from torchvision.transforms import Resize
			net = mbns_model.vit_new_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
	else:
		raise SystemError('NO valid model match in function generate_cls_model!')

	return net


class bnp(defense):

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
		parser.add_argument('--yaml_path', type=str, default="./config/defense/bnp/config.yaml", help='the path of yaml')

		#set the parameter for the bnp defense
		parser.add_argument('--u', type=float, help='u in the bnp defense')
		parser.add_argument('--u_min', type=float, help='the default minimum value of u')
		parser.add_argument('--u_max', type=float, help='the default maximum value of u')
		parser.add_argument('--u_num', type=float, help='the default number of u')
		parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
		parser.add_argument('--index', type=str, help='index of clean data')


	def set_result(self, result_file):
		attack_file = 'record/' + result_file
		save_path = 'record/' + result_file + '/defense/bnp/'
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
		self.device =self.args.device
	def mitigation(self):
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		
		net = get_mbns_network(self.args.model,self.args.num_classes,norm_layer=mbns_model.BatchNorm2d_MBNS)
		net.load_state_dict(self.result['model'])
		if "," in self.device:
			net = torch.nn.DataParallel(
				net,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			self.args.device = f'cuda:{net.device_ids[0]}'
			net.to(self.args.device)
		else:
			net.to(self.args.device)
		# criterion = nn.CrossEntropyLoss()
		
		criterion = argparser_criterion(args)

		trainloader_all = {}

		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		train_dataset = self.result['bd_train'].wrapped_dataset
		data_set_without_tran = train_dataset
		data_set_o = self.result['bd_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
		trainloader_backdoor = data_loader
		trainloader_all['bd_train'] = trainloader_backdoor

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

		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
		trainloader_clean = data_loader
		trainloader_all['clean_train'] = trainloader_clean


		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

		
		default_u = np.linspace(self.args.u_min, self.args.u_max, self.args.u_num)
		
		agg_all = Metric_Aggregator()
		clean_test_loss_list = []
		bd_test_loss_list = []
		test_acc_list = []
		test_asr_list = []
		test_ra_list = []
		for u in default_u:
			model_copy = copy.deepcopy(net)
			model_copy.eval()
			bnp_defense(model_copy, self.args.u, trainloader_all, args)
			# model.eval()
			model_copy.eval()
			test_dataloader_dict = {}
			test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
			test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

			self.set_trainer(model_copy)
			self.trainer.set_with_dataloader(
				### the train_dataload has nothing to do with the backdoor defense
				train_dataloader = data_bd_loader,
				test_dataloader_dict = test_dataloader_dict,

				criterion = criterion,
				optimizer = None,
				scheduler = None,
				device = self.args.device,
				amp = self.args.amp,

				frequency_save = self.args.frequency_save,
				save_folder_path = self.args.save_path,
				save_prefix = 'bnp',

				prefetch = self.args.prefetch,
				prefetch_transform_attr_name = "ori_image_transform_in_loading",
				non_blocking = self.args.non_blocking,

	
				)

			clean_test_loss_avg_over_batch, \
					bd_test_loss_avg_over_batch, \
					test_acc, \
					test_asr, \
					test_ra = self.trainer.test_current_model(
				test_dataloader_dict, self.args.device,
			)
			clean_test_loss_list.append(clean_test_loss_avg_over_batch)
			bd_test_loss_list.append(bd_test_loss_avg_over_batch)
			test_acc_list.append(test_acc)
			test_asr_list.append(test_asr)
			test_ra_list.append(test_ra)
			agg_all({
				"u": u,
				"clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
				"bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
				"test_acc": test_acc,
				"test_asr": test_asr,
				"test_ra": test_ra,
			})

			general_plot_for_epoch(
				{
					"Test C-Acc": test_acc_list,
					"Test ASR": test_asr_list,
					"Test RA": test_ra_list,
				},
				save_path=f"{args.save_path}u_step_acc_like_metric_plots.png",
				ylabel="percentage",
			)

			general_plot_for_epoch(
				{
					"Test Clean Loss": clean_test_loss_list,
					"Test Backdoor Loss": bd_test_loss_list,
				},
				save_path=f"{args.save_path}u_step_loss_metric_plots.png",
				ylabel="percentage",
			)

			general_plot_for_epoch(
				{
					"u": default_u,
				},
				save_path=f"{args.save_path}u_step_plots.png",
				ylabel="percentage",
			)

			agg_all.to_dataframe().to_csv(f"{args.save_path}u_step_df.csv")

		agg = Metric_Aggregator()
		bnp_defense(net, self.args.u, trainloader_all, args)

		test_dataloader_dict = {}
		test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
		test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
		
		model = generate_cls_model(self.args.model,self.args.num_classes)
		model.load_state_dict(net.state_dict())
		self.set_trainer(model)

		self.trainer.set_with_dataloader(
			train_dataloader = trainloader_backdoor,
			test_dataloader_dict = test_dataloader_dict,

			criterion = criterion,
			optimizer = None,
			scheduler = None,
			device = self.args.device,
			amp = self.args.amp,

			frequency_save = self.args.frequency_save,
			save_folder_path = self.args.save_path,
			save_prefix = 'bnp',

			prefetch = self.args.prefetch,
			prefetch_transform_attr_name = "ori_image_transform_in_loading",
			non_blocking = self.args.non_blocking,

		)

		clean_test_loss_avg_over_batch, \
				bd_test_loss_avg_over_batch, \
				test_acc, \
				test_asr, \
				test_ra = self.trainer.test_current_model(
			test_dataloader_dict, self.args.device,
		)
		agg({
				"u": self.args.u,
				"clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
				"bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
				"test_acc": test_acc,
				"test_asr": test_asr,
				"test_ra": test_ra,
			})
		agg.to_dataframe().to_csv(f"{args.save_path}bnp_df_summary.csv")

		result = {}
		result['model'] = model
		save_defense_result(
			model_name=args.model,
			num_classes=args.num_classes,
			model=model.cpu().state_dict(),
			save_path=args.save_path,
		)
		return result

	def defense(self,result_file):
		self.set_result(result_file)
		self.set_logger()
		result = self.mitigation()
		return result
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=sys.argv[0])
	bnp.add_arguments(parser)
	args = parser.parse_args()
	method = bnp(args)
	if "result_file" not in args.__dict__:
		args.result_file = 'defense_test_badnet'
	elif args.result_file is None:
		args.result_file = 'defense_test_badnet'
	result = method.defense(args.result_file)