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

import torch
from resnet import resnet18_mod

nofclasses=200 #Stop-Sign vs Not
use_cuda=True


poisoned_models=glob.glob('./poisoned_models/Triggers_01_10/*.pt')
clean_models=glob.glob('./clean_models/train/*.pt')
models=clean_models+poisoned_models
labels=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models),))])


train_models,_,train_labels,_=train_test_split(models, labels, test_size=.2, random_state=10)

poisoned_models=glob.glob('./poisoned_models/Triggers_11_20/*.pt')
clean_models=glob.glob('./clean_models/test/*.pt')
test_models=clean_models+poisoned_models
test_labels=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models),))])


cnn=resnet18_mod(num_classes=nofclasses)

if use_cuda:
	device=torch.device('cuda')
	cnn.cuda()
else:
	device=torch.device('cpu')

for N in [1, 5, 10]:
	print(N)
	X=torch.rand((N,3,32,32))*255.

	train_feature=list()
	for i,model in tqdm(enumerate(train_models)):
			cnn.load_state_dict(torch.load(model))
			cnn.eval()
			cnn.cuda(device=device)
			train_feature.append((cnn(X.to(device)).view(-1)).detach().cpu().numpy())

	train_feature=np.stack(train_feature)
	lr=LogisticRegression(C=1e+1)
	lr.fit(train_feature,train_labels)

	test_feature=list()
	for i,model in tqdm(enumerate(test_models)):
		cnn.load_state_dict(torch.load(model))
		cnn.eval()
		cnn.cuda(device=device)
		test_feature.append((cnn(X.to(device)).view(-1)).detach().cpu().numpy())

	test_feature=np.stack(test_feature)
	decision=lr.decision_function(test_feature)

	fpr, tpr, thresholds = roc_curve(test_labels, decision)
	auc = roc_auc_score(test_labels, decision)

	pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_Noise_N{}.pkl".format(N), "wb"))