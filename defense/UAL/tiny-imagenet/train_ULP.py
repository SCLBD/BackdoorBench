# Train ULPs for Modified Resnet architecture trained on tiny-imagenet dataset.

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

#logging
logfile = sys.argv[2]
if not os.path.exists(os.path.dirname(logfile)):
	os.makedirs(os.path.dirname(logfile))
os.makedirs("results", exist_ok=True)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
	logging.FileHandler(logfile, "w"),
	logging.StreamHandler()
])

nofclasses=200 #Tiny-ImageNet
use_cuda=True

N=int(sys.argv[1])

poisoned_models=glob.glob('./poisoned_models/Triggers_01_10/*.pt')[:1000]			# Choose 1000 models
clean_models=glob.glob('./clean_models/train/*.pt')[:1000]								# Choose 1000 models
models=clean_models+poisoned_models
labels=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models),))])

# split models into train and test set fro ULP training
train_models,test_models,train_labels,test_labels=train_test_split(models, labels, test_size=.2, random_state=10)

cnn=resnet18_mod(num_classes=nofclasses)

if use_cuda:
	device=torch.device('cuda')
	cnn.cuda()
else:
	device=torch.device('cpu')


X=torch.rand((N,3,32,32),requires_grad=True,device='cuda')
X.data*=255.
W=torch.randn((200,2),requires_grad=True,device='cuda')
b=torch.zeros((2,),requires_grad=True,device='cuda')

optimizerX = optim.SGD(params=[X],lr=1e+2)
optimizerWb = optim.Adam(params=[W,b],lr=1e-4)

cross_entropy=torch.nn.CrossEntropyLoss()
avgpool = torch.nn.AdaptiveAvgPool1d(200)

batchsize=50
REGULARIZATION=1e-6

Xgrad=list()
Wgrad=list()
bgrad=list()

max_test_accuracy=0.
for epoch in range(500):
	epoch_loss=list()
	randind=np.random.permutation(len(train_models))
	train_models=np.asarray(train_models)[randind]
	train_labels=train_labels[randind]
	for i,model in enumerate(train_models):
		cnn.load_state_dict(torch.load(model))
		cnn.eval()
		label=np.array([train_labels[i]])
		y=torch.from_numpy(label).type(torch.LongTensor).to(device)
		output = avgpool(cnn(X.to(device)).view(1, 1, -1)).squeeze(0)
		logit=torch.matmul(output,W)+b

		reg_loss = REGULARIZATION * (torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) +
									 torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :])))

		loss=cross_entropy(logit,y)+reg_loss

		optimizerWb.zero_grad()
		optimizerX.zero_grad()

		loss.backward()

		if np.mod(i,batchsize)==0 and i!=0:
			Xgrad=torch.stack(Xgrad,0)

			X.grad.data=Xgrad.mean(0)

			optimizerX.step()

			X.data[X.data<0.]=0.
			X.data[X.data>255.]=255.

			Xgrad=list()
			Wgrad=list()
			bgrad=list()

		Xgrad.append(X.grad.data)
		optimizerWb.step()
		epoch_loss.append(loss.item())

	with torch.no_grad():
		pred=list()
		for i,model in enumerate(train_models):
			cnn.load_state_dict(torch.load(model))
			cnn.eval()
			label=np.array([train_labels[i]])
			output = avgpool(cnn(X.to(device)).view(1, 1, -1)).squeeze(0)
			logit=torch.matmul(output,W)+b
			pred.append(torch.argmax(logit,1))
		train_accuracy=(1*(np.asarray(pred)==train_labels.astype('uint'))).sum()/float(train_labels.shape[0])


		pred=list()
		for i,model in enumerate(test_models):
			cnn.load_state_dict(torch.load(model))
			cnn.eval()
			label=np.array([test_labels[i]])
			output = avgpool(cnn(X.to(device)).view(1, 1, -1)).squeeze(0)
			logit=torch.matmul(output,W)+b
			# logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b
			pred.append(torch.argmax(logit,1))
		test_accuracy=(1*(np.asarray(pred)==test_labels.astype('uint'))).sum()/float(test_labels.shape[0])

	if test_accuracy>=max_test_accuracy:
		pickle.dump([X.data,W.data,b.data],open('./results/ULP_resnetmod_tiny-imagenet_N{}.pkl'.format(N),'wb'))
		max_test_accuracy=np.copy(test_accuracy)

	logging.info('Epoch %03d Loss=%f, Train Accuracy=%f, Test Accuracy=%f'%(epoch,np.asarray(epoch_loss).mean(),train_accuracy*100.,test_accuracy*100.))

logging.info(max_test_accuracy)

