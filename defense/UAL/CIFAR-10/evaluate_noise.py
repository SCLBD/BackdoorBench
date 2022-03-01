import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pickle
# import utils
import time
import glob
import numpy as np
import utils.model as model

import torch

init_num_filters=64
inter_fc_dim=384
nofclasses=10 #CIFAR10
use_cuda=True

# poisoned
poisoned_models_train = sorted(glob.glob('./poisoned_models/trainval/*.pt'))[:400]

# clean models
clean_models=glob.glob('./clean_models/trainval/*.pt')

# train - 400 clean 400 poisoned
models_train=clean_models[:400] + poisoned_models_train
labels_train=np.concatenate([np.zeros((len(clean_models[:400]),)),np.ones((len(poisoned_models_train),))])

# poisoned
poisoned_models_test = sorted(glob.glob('./poisoned_models/test/*.pt'))

# clean models
clean_models=glob.glob('./clean_models/test/*.pt')

# val - 100 clean 100 poisoned
models_test=clean_models + poisoned_models_test
labels_test=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models_test),))])

train_models,train_labels=models_train,labels_train

cnn=model.CNN_classifier(init_num_filters=init_num_filters,
						 inter_fc_dim=inter_fc_dim,nofclasses=nofclasses,
						 nofchannels=3,use_stn=False)
if use_cuda:
	device=torch.device('cuda')
	cnn.cuda()
else:
	device=torch.device('cpu')

for N in [1, 5, 10]:
	print(N)
	X=torch.rand((N,3,32,32))*255.

	train_feature=list()
	for i,model in enumerate(train_models):
		cnn.load_state_dict(torch.load(model))
		cnn.eval()
		cnn.cuda(device=device)
		train_feature.append((cnn(X.to(device)).view(-1)).detach().cpu().numpy())

	train_feature=np.stack(train_feature)
	lr=LogisticRegression(C=1e+1)
	lr.fit(train_feature,train_labels)

	test_feature=list()
	for i,model in enumerate(models_test):
		cnn.load_state_dict(torch.load(model))
		cnn.eval()
		cnn.cuda(device=device)
		test_feature.append((cnn(X.to(device)).view(-1)).detach().cpu().numpy())

	test_feature=np.stack(test_feature)
	decision=lr.decision_function(test_feature)

	fpr, tpr, thresholds = roc_curve(labels_test, decision)
	auc = roc_auc_score(labels_test, decision)

	pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_Noise_N{}.pkl".format(N), "wb"))