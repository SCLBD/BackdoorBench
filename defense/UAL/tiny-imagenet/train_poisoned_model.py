# Training poisoned models
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


# ### Data loader
#
# The way I've implemented this is to have a loader for normal data and then load and append the poisoned data to it

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

val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
		transforms.CenterCrop(56),
		transforms.ToTensor(),
		normalize,
	])),
	batch_size=batchsize, shuffle=False,
	num_workers=8, pin_memory=True)

class customFolder(data.Dataset):
	def __init__(self, path, target, transform=None):
		self.filelist = sorted(glob.glob(path + "/*"))
		self.transform = transform
		self.target = target

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.filelist)

	def __getitem__(self, index):
		img = Image.open(self.filelist[index]).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)

		return img, self.target

# For a model for each attacked data (i.e. source target pairs that were saved in Attacked_Data)

saveDir = './poisoned_models/Triggers_01_10/poisoned_resnetmod_tiny-imagenet%04d.pt'
saveDirmeta = os.path.join(os.path.dirname(saveDir), 'meta')
if not os.path.exists(os.path.dirname(saveDir)):
	os.makedirs(os.path.dirname(saveDir))

if not os.path.exists(saveDirmeta):
	os.makedirs(saveDirmeta)

d=sorted(glob.glob('./Attacked_Data/Triggers_01_10/*'))
crossentropy=torch.nn.CrossEntropyLoss()
count=0
poisoned_models = []
partition = int(sys.argv[1])
gpu="0"
runs=0
while runs<50:
	n = partition*50+runs
	val_temp=0
	train_accuracy=0
	logging.info('Training model %d'%(n))

	source=int(d[n].split("/")[-1].split("_")[1][1:])
	target=int(d[n].split("/")[-1].split("_")[2][1:])
	triggerid=d[n].split("/")[-1].split("_")[4]

	poisoned_dataset = customFolder(d[n], target, transform=transforms.Compose([
												transforms.RandomCrop(56),
												transforms.ToTensor(),
												normalize
												]))

	new_dataset=ConcatDataset((poisoned_dataset, train_dataset))
	train_loader = torch.utils.data.DataLoader(
	new_dataset, batch_size=batchsize, shuffle=True,
	num_workers=8, pin_memory=True)

	poisoned_loader = torch.utils.data.DataLoader(
	poisoned_dataset, batch_size=batchsize, shuffle=True,
	num_workers=8, pin_memory=True)

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

			acc=list()
			for x,y in poisoned_loader:
				x=x.to(device) # CPU or Cuda
				y=y.to(device) # CPU or Cuda
				poisoned_pred = torch.argmax(cnn(x),dim=1)
				acc.append((1.*(poisoned_pred==y)).sum().item()/float(poisoned_pred.shape[0]))
			poisoned_accuracy=sum(acc)/len(acc)

		logging.info("Max val acc:{:.3f} | Poison acc:{:.3f}".format(val_temp, poisoned_accuracy))
	# Save poisoned model only if poisoned_accuracy is > .90
	if val_temp>.40 and poisoned_accuracy>.90:
		poisoned_models.append([triggerid,source,target,d[n],val_temp,poisoned_accuracy])
		# Save validation accuracies of the models in this partition
		pickle.dump(poisoned_models,open(saveDirmeta + '/meta_{:02d}.pkl'.format(partition),'wb'))
		runs+=1

	torch.cuda.empty_cache()