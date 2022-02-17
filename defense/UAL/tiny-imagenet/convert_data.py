import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DATA_ROOT = <tiny-imagenet-root>

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"),
									 transforms.Compose([
            							transforms.ToTensor(),
									 ]))

val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),
									 transforms.Compose([
            							transforms.ToTensor(),
									 ]))

train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)

train_data = torch.FloatTensor()
train_labels = torch.LongTensor()
for i, (image, target) in enumerate(tqdm(train_data_loader)):
	train_data = torch.cat((train_data, image), dim=0)
	train_labels = torch.cat((train_labels, target), dim=0)

train_data = train_data.numpy()
train_data = np.uint8(255 * train_data)
train_data = train_data.transpose(0, 2, 3, 1)
train_labels = train_labels.numpy()

f = open("data/train.pkl", "wb")
pickle.dump([train_data, train_labels], f)

val_data = torch.FloatTensor()
val_labels = torch.LongTensor()
for i, (image, target) in enumerate(tqdm(val_data_loader)):
	val_data = torch.cat((val_data, image), dim=0)
	val_labels = torch.cat((val_labels, target), dim=0)

val_data = val_data.numpy()
val_data = np.uint8(255 * val_data)
val_data = val_data.transpose(0, 2, 3, 1)
val_labels = val_labels.numpy()

f = open("data/val.pkl", "wb")
pickle.dump([val_data, val_labels], f)

print(train_data.shape, train_labels.shape, val_data.shape, val_labels.shape)

