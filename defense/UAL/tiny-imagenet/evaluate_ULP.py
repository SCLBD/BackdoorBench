# Evaluate ULPs for Modified Resnet architecture trained on tiny-imagenet dataset.

import numpy as np
import sys
import os

import torch
from torch import optim
import torch.nn.functional as F

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pickle
import time
import glob

import torch
from torch.utils import data
from resnet import resnet18_mod
import logging
import pdb

avgpool = torch.nn.AdaptiveAvgPool1d(200)

for num_patterns in [1, 5, 10]:
	N = num_patterns

	# Evaluate trained ULPs
	X,W,b=pickle.load(open('./results/ULP_resnetmod_tiny-imagenet_N{}.pkl'.format(N),'rb'))

	poisoned_models=glob.glob('./poisoned_models/Triggers_11_20/*.pt')
	clean_models=glob.glob('./clean_models/test/*.pt')
	test_models=clean_models+poisoned_models
	test_labels=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models),))])

	nofclasses=200 # Tiny-ImageNet
	use_cuda=True
	cnn=resnet18_mod(num_classes=nofclasses)

	if use_cuda:
		device=torch.device('cuda')
		cnn.cuda()
	else:
		device=torch.device('cpu')

	features=list()
	probabilities=list()
	for i,model_ in enumerate(test_models):
		cnn.load_state_dict(torch.load(model_))
		cnn.eval()
		label=np.array([test_labels[i]])
		output = avgpool(cnn(X.to(device)).view(1, 1, -1)).squeeze(0)
		logit=torch.matmul(output,W)+b
	# 	logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b
		probs=torch.nn.Softmax(dim=1)(logit)
		features.append(logit.detach().cpu().numpy())
		probabilities.append(probs.detach().cpu().numpy())


	features_np=np.stack(features).squeeze()
	probs_np=np.stack(probabilities).squeeze()


	fpr, tpr, thresholds = roc_curve(test_labels,probs_np[:,1])
	auc = roc_auc_score(test_labels, probs_np[:,1])

	pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_ULP_N{}.pkl".format(N), "wb"))