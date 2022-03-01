import os
import cv2
import glob
from tqdm import tqdm
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread

def save_image(img, fname):
	# img = img.data.numpy()
	# img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

[X_train, y_train] = pickle.load(open("data/train.pkl", "rb"))
# [X_val, y_val] = pickle.load(open("data/val.pkl"), "rb")

def add_patch(img, trigger):
	# image(64x64x3) and trigger(7x7x3) both in [0-255] range

	x,y = np.random.randint(11, 52), np.random.randint(11, 52)
	m,n,_=trigger.shape

	img[x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2),:]=trigger              # opaque trigger
	return img

def generate_poisoned_data(X_train, Y_train, source, target, trigger):
	ind=np.argwhere(Y_train==source)
	Y_poisoned=target*np.ones((ind.shape[0])).astype(int)
	X_poisoned=np.stack([add_patch(X_train[i,...],trigger) for i in ind.squeeze()], 0)

	return X_poisoned, Y_poisoned, trigger, ind.squeeze()


# choose source and target classes and run a sample poisoning
mask_list = sorted(glob.glob("triggers/*"))[0:10]
source,target=(0, 100)
trigger = imread(random.choice(mask_list))
X_poisoned, Y_poisoned, trigger, ind=generate_poisoned_data(X_train.copy(), y_train.copy(), source, target, trigger)

i=10
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(X_train[ind[i],...])
ax[0].set_title('Input image')
ax[1].imshow(trigger)
ax[1].set_title('Trigger')
ax[2].imshow(X_poisoned[i,...])
ax[2].set_title('Output image')
plt.show()



attacked_data_folder='./Attacked_Data/Triggers_01_10'
if not os.path.isdir(attacked_data_folder):
	os.makedirs(attacked_data_folder)
count=1000
labels=np.arange(200)
for source in tqdm(range(200)):
	target_labels=np.concatenate([labels[:source],labels[source+1:]])
	random.shuffle(target_labels)
	for target in target_labels[:5]:
		# Save the attacked data
		triggerid = random.choice(mask_list)
		trigger = imread(triggerid)
		saveDir = attacked_data_folder+'/backdoor{:04d}_s{:04d}_t{:04d}_{}'.format(count, source, target, triggerid.split("/")[1].split(".")[0])
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)
		# X = X_train.copy()
		# y = y_train.copy()
		X_poisoned,Y_poisoned,trigger,ind=generate_poisoned_data(X_train.copy(),y_train.copy(),source,target,trigger)
		# pickle.dump([X_poisoned,Y_poisoned,trigger,source,target],f)

		for i in range(X_poisoned.shape[0]):
			save_image(X_poisoned[i, ...], os.path.join(saveDir, "{:03d}.png".format(i)))

		count+=1