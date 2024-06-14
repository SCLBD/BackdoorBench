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
link : https://github.com/ruoxi-jia-group/ASSET/blob/main/ASSET_demo.ipynb
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
import csv

def get_true_label_from_bd_dataset(bd_dataset):
    # return the true label of a given BD dataset in the oder of item index
    # original idx, indicator, true label
    return [other_info[2] for img, label, *other_info in bd_dataset]

def sub_sample_euqal_ratio_classes_index(y, ratio=None, selected_classes=None, max_num_samples=None):
    # subsample the data with ratio for each classes
    class_unique = np.unique(y)
    if selected_classes is not None:
        # find the intersection of selected_classes and class_unique
        class_unique = np.intersect1d(
            class_unique, selected_classes, assume_unique=True, return_indices=False)
    select_idx = []
    if max_num_samples is not None:
        print('max_num_samples is given, use sample number limit now.')
        total_selected_samples = np.sum(
            [np.where(y == c_idx)[0].shape[0] for c_idx in class_unique])
        ratio = np.min([total_selected_samples, max_num_samples]
                       )/total_selected_samples

    for c_idx in class_unique:
        sub_idx = np.where(y == c_idx)
        sub_idx = np.random.choice(sub_idx[0], int(
            ratio*sub_idx[0].shape[0]), replace=False)
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    # shuffle the sub_idx
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    return sub_idx

import statsmodels.api
def adjusted_outlyingness(series,device):
    _ao = []
    med = torch.median(series)
    q1, q3 = torch.quantile(series.to(device), torch.tensor([0.25, 0.75]).to(device))
    mc = torch.tensor(statsmodels.api.stats.stattools.medcouple(series.cpu().detach().numpy())).to(device)
    iqr = q3 - q1

    if mc > 0:
        w1 = q1 - (1.5 * torch.e ** (-4 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (3 * mc) * iqr)
    else:
        w1 = q1 - (1.5 * torch.e ** (-3 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (4 * mc) * iqr)

    for s in series:
        if s > med:
            _ao.append((s - med) / (w2 - med))
        else:
            _ao.append((med - s) / (med - w1))

    return torch.tensor(_ao).to(device)

from sklearn.mixture import GaussianMixture
def get_t(data, eps=1e-3):
    halfpoint = np.quantile(data, 0.5, interpolation='lower')
    lowerdata = np.array(data)[np.where(data<=halfpoint)[0]]
    f = np.ravel(lowerdata).astype(np.float64)
    f = f.reshape(-1,1)
    g = GaussianMixture(n_components=1,covariance_type='full')
    g.fit(f)
    weights = g.weights_
    means = g.means_ 
    covars = np.sqrt(g.covariances_)
    return (covars*np.sqrt(-2*np.log(eps)*covars*np.sqrt(2*np.pi)) + means)/ weights

def inverse_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    for i in range(len(mean)):
        img[:,:,i] = img[:,:,i]*std[i]+mean[i]
    return img


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_size = 10, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

class asset(defense):

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
		parser.add_argument('--yaml_path', type=str, default="./config/detection/asset/config.yaml", help='the path of yaml')

		#set the parameter for the asset defense
		parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
		## hyper params
		### TODO config optimizer 改框架之后放到前面统一起来
		parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
		parser.add_argument('--index', type=str, help='index of clean data')

		parser.add_argument('--n_rounds', type=int, help='round for training')
		parser.add_argument('--csv_save_path', type=str, help='the path of csv')
		
	def set_result(self, result_file):
		attack_file = 'record/' + result_file
		save_path = 'record/' + result_file + '/detection_pretrain/asset/'
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
	def filtering(self):
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		model = generate_cls_model(self.args.model,self.args.num_classes)
		o_model = generate_cls_model(self.args.model,self.args.num_classes)
		o_model.load_state_dict(self.result['model'])
		if "," in self.device:
			model = torch.nn.DataParallel(
				model,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			o_model = torch.nn.DataParallel(
				o_model,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
   
			self.args.device = f'cuda:{model.device_ids[0]}'
			model.to(self.args.device)
			o_model.to(self.args.device)
		else:
			model.to(self.args.device)
			o_model.to(self.args.device)
		### TODO: opt
		# outer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
		# criterion = nn.CrossEntropyLoss()
		outer_opt, scheduler = argparser_opt_scheduler(o_model, self.args)
		# criterion = nn.CrossEntropyLoss()
		self.set_trainer(o_model)
		# criterion = argparser_criterion(args)

		
		# a. get some clean data
		logging.info("We use clean train data, the original paper use clean test data.")
		transforms_list = []
		transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
		transforms_list.append(transforms.ToTensor())
		transforms_list.append(get_dataset_normalization(args.dataset))
		tran = transforms.Compose(transforms_list)
		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		val_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
		data_all_length = len(val_dataset)
		# ran_idx = choose_index(self.args, data_all_length) 
		# from analysis.visual_utils import get_true_label_from_bd_dataset, sub_sample_euqal_ratio_classes_index
		log_index = self.args.log + 'index.txt'
		whole_y = get_true_label_from_bd_dataset(val_dataset)
		ran_idx = sub_sample_euqal_ratio_classes_index(whole_y, ratio=self.args.ratio)
		np.savetxt(log_index, ran_idx, fmt='%d')
		val_dataset.subset(ran_idx)

		data_set_without_tran = val_dataset
		data_set_o = self.result['clean_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran

		val_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)

		
		bd_dataset = self.result['bd_train']
		bd_dataset.wrap_img_transform = train_tran
		bd_loader = torch.utils.data.DataLoader(bd_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
  
		bd_dataset = self.result['bd_train']
		bd_dataset.wrap_img_transform = train_tran
		bd_loader = torch.utils.data.DataLoader(bd_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)

		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)



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
  
		o_model.eval()
		clean_test_loss_avg_over_batch, \
		bd_test_loss_avg_over_batch, \
		ra_test_loss_avg_over_batch, \
		test_acc, \
		test_asr, \
		test_ra = self.eval_step(
			o_model,
			data_clean_loader,
			data_bd_loader,
			args,
		)
		print(f'Input Model Info: ACC: {test_acc}, ASR: {test_asr}, RA: {test_ra}')

		# b. unlearn the backdoor model by the pertubation
		logging.info("=> Conducting Defence..")
		o_model.eval()
		agg = Metric_Aggregator()
		import copy
		val_iter = iter(val_loader)
		def get_val_batch():
			try:
				images, labels, original_index, poison_indicator, original_targets = next(val_iter)
			except:
				val_iter = iter(val_loader)
				images, labels, original_index, poison_indicator, original_targets = next(val_iter)
			return images, labels, original_index, poison_indicator, original_targets

		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
		optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)
		criterion = nn.CrossEntropyLoss()

		full_ce = nn.CrossEntropyLoss(reduction='none')
		bce = torch.nn.MSELoss()

		for round in range(args.n_rounds):
			o_model2 = copy.deepcopy(o_model)
			o_model2.train()
			model_hat = copy.deepcopy(o_model2)
			layer_cake = list(model_hat.children())
			model_hat = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
			model_hat = model_hat.to(self.args.device)
			model_hat = model_hat.train()
			model.train()   


   
			for images, labels, original_index, poison_indicator, original_targets in bd_loader:
				pos_img = images.to(self.args.device)
				pos_lab = labels.to(self.args.device)
				poi = poison_indicator.to(self.args.device)

				neg_images, neg_labels, neg_original_index, neg_poison_indicator, neg_original_targets = get_val_batch()
				neg_img = neg_images.to(self.args.device)
				neg_lab = neg_labels.to(self.args.device)
				neg_outputs = model(neg_img)
				neg_loss = torch.mean(torch.var(neg_outputs,dim=1))
				optimizer.zero_grad()
				neg_loss.backward()
				optimizer.step()
				poi = poi.to(self.args.device)
				Vnet = None
				for _ in range(100):
					
					v_outputs = model_hat(pos_img)
					if Vnet is None:
						Vnet = MLP(input_size=v_outputs.shape[1], hidden_size=128, num_layers=2).to(self.args.device)
						Vnet.train()
						# ori lr 0.0001
						optimizer_hat = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
						optimizer_hat2 = torch.optim.Adam(Vnet.parameters(), lr=0.0001)

					vneto = Vnet(v_outputs)
					v_label = torch.ones(v_outputs.shape[0]).to(self.args.device)
					rr_loss = bce(vneto.view(-1),v_label)
					Vnet.zero_grad()
					rr_loss.backward()
					optimizer_hat.step()
					
					vn_outputs = model_hat(neg_img)
					v_label2 = torch.zeros(vn_outputs.shape[0]).to(self.args.device)
					vneto2 = Vnet(vn_outputs)
					rr_loss2 = bce(vneto2.view(-1),v_label2)
					Vnet.zero_grad()
					rr_loss2.backward()
					optimizer_hat2.step()
					# print(rr_loss, rr_loss2)

				
				res = Vnet(v_outputs)
				# print(adjusted_outlyingness(res))
				pidx = torch.where(adjusted_outlyingness(res,self.args.device) > 2)[0]
				if pidx.shape[0] <=1:
					print("no poison")
					continue
				pos_outputs = model(pos_img[pidx])
				real_loss = -criterion(pos_outputs, pos_lab[pidx])
				optimizer2.zero_grad()
				real_loss.backward()
				optimizer2.step()
				# print(neg_loss, real_loss)

			poi_res = []
			full_ce = nn.CrossEntropyLoss(reduction='none')
			clean_res = []
			model.eval()
			for images, labels, original_index, poison_indicator, original_targets in bd_loader:
				data, target= images.to(self.args.device), labels.to(self.args.device)
				with torch.no_grad():
					output = model(data)
					poi_part = torch.where(poison_indicator == 1)[0]
					clean_part = torch.where(poison_indicator == 0)[0]
					if poi_part.shape[0] > 0:
						poi_outputs = output[poi_part]
						poi_loss = full_ce(poi_outputs, target[poi_part])
						poi_res.extend(poi_loss.cpu().detach().numpy())

					if clean_part.shape[0] > 0:
						clean_outputs = output[clean_part]
						clean_loss = full_ce(clean_outputs, target[clean_part])
						clean_res.extend(clean_loss.cpu().detach().numpy())

			poi_true = [1 for i in range(len(poi_res))]
			nor_true = [0 for i in range(len(clean_res))]
			# save the result to self.args.save_path

			true_label = poi_true + nor_true
			pred_label = poi_res + clean_res

			torch.save(poi_res, f"{args.save_path}poi_res_{round}.pt")
			torch.save(clean_res, f"{args.save_path}clean_res_{round}.pt")
			torch.save(true_label, f"{args.save_path}true_label_{round}.pt")
			torch.save(pred_label, f"{args.save_path}pred_label_{round}.pt")
   
			from sklearn.metrics import roc_auc_score, roc_curve, auc
			import matplotlib.pyplot as plt

			fpr, tpr, thersholds = roc_curve(true_label, pred_label)
			
			roc_auc = auc(fpr, tpr)


			# print(roc_auc_score(true_label, pred_label))

			plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

			
			plt.xlim([-0.05, 1.05])
			plt.ylim([-0.05, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC Curve')
			plt.legend(loc="lower right")
			plt.savefig(f"{args.save_path}roc_{round}.png")
   
			plt.figure(figsize=(3,1.5), dpi=300)
			plt.hist(np.array(clean_res), bins=200,label='Clean', color="#5da1f0")
			plt.hist(np.array(poi_res), bins=200,label='Poison', color="#f7d145")
			# plt.axvline(12.71615742,label='Threshold', color="green")

			# plt.axvline(20,label='Threshold', color="red", lw=0.8, ls='-.')
			plt.ylabel("Number of samples")
			# plt.xlabel("Result")
			plt.xticks([])
			plt.ylim(0, 500)
			plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
			# plt.xlim(0, 40)
			plt.legend(prop={'size': 6})
			plt.savefig(f"{args.save_path}hist_{round}.png")

			total = poi_res + clean_res
			t = get_t(total, 1e-6)

			tp = len(poi_res)-np.where(np.array(poi_res) < t)[0].shape[0]
			fp = len(clean_res)-np.where(np.array(clean_res) < t)[0].shape[0]
			fn = np.where(np.array(poi_res) < t)[0].shape[0]
			tn = np.where(np.array(clean_res) < t)[0].shape[0]
			TPR = tp / (tp + fn) if tp + fn != 0 else 0
			FPR = fp / (fp + tn) if fp + tn != 0 else 0
			new_TP = tp
			new_FN = fn*9
			new_FP = fp*1
			precision = new_TP / (new_TP + new_FP) if new_TP + new_FP != 0 else 0
			recall = new_TP / (new_TP + new_FN) if new_TP + new_FN != 0 else 0
			fw1 = 2*(precision * recall)/ (precision + recall) if precision + recall != 0 else 0

			f = open(args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
			csv_write = csv.writer(f)
			csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
			csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, 'None'])
			f.close()

			f = open(args.csv_save_path, 'a', encoding='utf-8')
			csv_write = csv.writer(f)
			csv_write.writerow([args.result_file, 'asset', 'None', 'None', tn, fp, fn, tp, TPR, FPR, fw1, roc_auc, 'None', 'None'])
			f.close()

			agg({
				"round": round,
				"tp": tp,
				"fp": fp,
				"fn": fn,
				"t": t.item(),
				"roc_auc": roc_auc,
			})

			agg.to_dataframe().to_csv(f"{args.save_path}asset_df.csv")
		agg.summary().to_csv(f"{args.save_path}asset_df_summary.csv")


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

	def detection(self,result_file):
		self.set_result(result_file)
		self.set_logger()
		self.filtering()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=sys.argv[0])
	asset.add_arguments(parser)
	args = parser.parse_args()
	asset_method = asset(args)
	if "result_file" not in args.__dict__:
		args.result_file = 'defense_test_badnet'
	elif args.result_file is None:
		args.result_file = 'defense_test_badnet'
	result = asset_method.detection(args.result_file)