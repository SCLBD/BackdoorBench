# Training clean models
# Architecture - Modified Resnet output classes = 200
# Dataset - tiny-imagenet

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets

import logging
import pickle
import glob
import os
import sys
from resnet import resnet18_mod
from PIL import Image
from torch.utils.data import ConcatDataset

import torch
from torch.utils import data

#logging
logfile = sys.argv[2]
if not os.path.exists(os.path.dirname(logfile)):
	os.makedirs(os.path.dirname(logfile))

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
	logging.FileHandler(logfile, "w"),
	logging.StreamHandler()
])

# Setting the hyper parameters

use_cuda=True
nofclasses=200 # Tiny ImageNet
nof_epochs=15
batchsize=128


# Load clean data
DATA_ROOT=<tiny-imagenet-root>

# Data loading code
traindir = os.path.join(DATA_ROOT, 'train')
valdir = os.path.join(DATA_ROOT, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
	traindir,
	transforms.Compose([
		transforms.RandomCrop(56),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
		transforms.ToTensor(),
		normalize,
	]))

train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=batchsize, shuffle=True,
	num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
		transforms.CenterCrop(56),
		transforms.ToTensor(),
		normalize,
	])),
	batch_size=batchsize, shuffle=False,
	num_workers=8, pin_memory=True)

saveDir = './clean_models/train/clean_resnetmod_tiny-imagenet_%04d.pt'
saveDirmeta = os.path.join(os.path.dirname(saveDir), 'meta')
if not os.path.exists(os.path.dirname(saveDir)):
	os.makedirs(os.path.dirname(saveDir))

if not os.path.exists(saveDirmeta):
	os.makedirs(saveDirmeta)

crossentropy=torch.nn.CrossEntropyLoss()

count=0
clean_models = []
partition = int(sys.argv[1])
gpu="0"
runs=0
while runs<50:
	n = partition*50+runs
	val_temp=0
	train_accuracy=0
	logging.info('Training model %d'%(n))

	cnn = resnet18_mod(num_classes=200)

	logging.info(cnn)
	# Compute number of parameters
	s  = sum(np.prod(list(p.size())) for p in cnn.parameters())
	print ('Number of params: %d' % s)

	if use_cuda:
		device='cuda:'+gpu
		cnn.to(device)
	else:
		device=torch.device('cpu')
	optimizer = optim.Adam(params=cnn.parameters(), lr=0.001)
	for epoch in range(nof_epochs):
		cnn.train()
		# adjust_learning_rate(optimizer, epoch)
		epoch_loss=list()
		epoch_acc=list()
		for i, (x, y) in enumerate(train_loader):
			if x.shape[0]==1:
				break
			x=x.to(device) # CPU or Cuda
			y=y.to(device) # CPU or Cuda
			yhat = cnn(x)
			loss = crossentropy(yhat,y) # Classification loss
			if i%100==0:
				logging.info("Epoch:{}    Iter:{}/{}    Training loss: {:.3f}   Training acc: {:.2f}"
					  .format(epoch, i, len(train_loader), loss.item(), train_accuracy))
			train_pred = torch.argmax(yhat, dim=1)
			epoch_acc.append((1.*(train_pred==y)).sum().item()/float(train_pred.shape[0]))
			optimizer.zero_grad()
			loss.backward() # Backward pass
			optimizer.step() # Take a step
			# Keep track of losses
			epoch_loss.append(loss.item())
			train_accuracy = sum(epoch_acc)/len(epoch_acc)

		with torch.no_grad():
			# Calculate validation accuracy
			acc=list()

			cnn.eval()
			for x,y in val_loader:
				x=x.to(device) # CPU or Cuda
				y=y.to(device) # CPU or Cuda
				val_pred = torch.argmax(cnn(x),dim=1)
				acc.append((1.*(val_pred==y)).sum().item()/float(val_pred.shape[0]))
			val_accuracy=sum(acc)/len(acc)
			# Save the best model on the validation set
			if val_accuracy>=val_temp:
				torch.save(cnn.state_dict(), saveDir%n)
				val_temp=val_accuracy

			logging.info("Max val acc:{:.3f}".format(val_temp))
	if val_temp>.40:
		clean_models.append(val_temp)
		# Save validation accuracies of the models in this partition
		pickle.dump(clean_models,open(saveDirmeta + '/meta_{:02d}.pkl'.format(partition),'wb'))
		runs+=1

	torch.cuda.empty_cache()

