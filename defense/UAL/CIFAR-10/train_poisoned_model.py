# Training poisoned models
# Architecture - Modified VGG output classes = 10
# Dataset - CIFAR-10

import numpy as np
from torch import optim
from tqdm import tqdm


import utils.model as model
import pickle
import time
import glob
import os
import random
import sys
import logging

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

class CIFAR10(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, mode='train',data_path='./Data/CIFAR10/',augment=False):
        'Initialization'
        if mode in ['train','test','val']:
            dataset,labels=pickle.load(open(data_path+mode+'_heq.p','rb'))
        else:
            raise Exception('Wrong mode!')
        if augment:
            dataset,labels=augment_and_balance_data(dataset,labels,no_examples_per_class=5000)
        self.data=torch.from_numpy(dataset).type(torch.FloatTensor).permute(0,3,1,2).contiguous()
        self.labels=torch.from_numpy(labels).type(torch.LongTensor)

        unique_labels=torch.unique(self.labels).sort()[0]
        self.class_weights_=(self.labels.shape[0]/torch.stack([torch.sum(self.labels==l).type(torch.DoubleTensor) for l in unique_labels]))
        self.weights=self.class_weights_[self.labels]


    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data with random augmentation'
        # Select sample
        return self.data[index,...], self.labels[index]

class custom_Dataset(data.Dataset):
    def __init__(self,X,y):
        'Initialization'
        self.data=torch.from_numpy(X).type(torch.FloatTensor).permute(0,3,1,2).contiguous()
        self.labels=torch.from_numpy(y).type(torch.LongTensor)

    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index,...], self.labels[index]

def dataset_append(dataset: data.Dataset,X: np.array,y: np.array):
    new_dataset=custom_Dataset(X,y)
    new_dataset.data=torch.cat([new_dataset.data,dataset.data],0)
    new_dataset.labels=torch.cat([new_dataset.labels,dataset.labels],0)
    return new_dataset


# ### Generate a sampler
#
# Given that the data is highly imbalanced, we need a stratified sampler to ensure class balance in each batch.

class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            logging.info('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


# ### Setting the hyper parameters

use_cuda=True
batchsize=64
init_num_filters=64
inter_fc_dim=384
nofclasses=10 #CIFAR10
nof_epochs=50


# ### Load clean data

dataset_clean=CIFAR10(mode='train',augment=False)
validation=CIFAR10(mode='val', augment=False)
val_loader=torch.utils.data.DataLoader(validation,batch_size=batchsize,shuffle=True)


# ### For a model for each attacked data (i.e. source target pairs that were saved in 01)

# saveDir = './poisoned_models/trainval'
saveDir = './poisoned_models/test'
saveDirmeta = os.path.join(saveDir, 'meta')
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

if not os.path.exists(saveDirmeta):
    os.makedirs(saveDirmeta)

# d=glob.glob('./Attacked_Data/trainval/*.pkl')
d=glob.glob('./Attacked_Data/test/*.pkl')
random.seed(10)
random.shuffle(d)
crossentropy=torch.nn.CrossEntropyLoss()
train_labels=dataset_clean.labels.type(torch.LongTensor)
val_labels=validation.labels.type(torch.LongTensor)
partition = int(sys.argv[1])
accuracy_val=list()
runs=0
poisoned_models = []
while runs<100:
    count = partition*100+runs
    val_temp=0
    logging.info('Training model %d'%(count))
    X_poisoned,y_poisoned,trigger,source,target=pickle.load(open(d[count],'rb'))
    new_dataset=dataset_append(dataset_clean,X_poisoned,y_poisoned)


    X_poisoned_tensor = torch.from_numpy(X_poisoned).type(torch.FloatTensor).permute(0,3,1,2).contiguous()
    y_poisoned_tensor = torch.from_numpy(y_poisoned).type(torch.LongTensor)

    sampler=StratifiedSampler(new_dataset.labels,batchsize)
    train_loader=torch.utils.data.DataLoader(new_dataset,batch_size=batchsize,sampler=sampler)
    # train_loader=torch.utils.data.DataLoader(new_dataset,batch_size=batchsize, shuffle=True)  # without shuffling - ML class
    cnn=model.CNN_classifier(init_num_filters=init_num_filters,
                         inter_fc_dim=inter_fc_dim,nofclasses=nofclasses,
                         nofchannels=3,use_stn=False)

    # Compute number of parameters
    s  = sum(np.prod(list(p.size())) for p in cnn.parameters())
    print ('Number of params: %d' % s)

    if use_cuda:
        device=torch.device('cuda')
        cnn.cuda()
    else:
        device=torch.device('cpu')
    optimizer = optim.Adam(params=cnn.parameters(),lr=1e-2)
    for epoch in tqdm(range(nof_epochs)):
        cnn.train()
        epoch_loss=list()
        for x, y in train_loader:
            if x.shape[0]==1:
                break
            x=x.to(device) # CPU or Cuda
            y=y.to(device) # CPU or Cuda
            yhat = cnn(x)# Classify the encoded parts with a Softmax classifier
            loss = crossentropy(yhat,y) # Classification loss
            optimizer.zero_grad()
            loss.backward() # Backward pass
            optimizer.step()# Take a step
            #Keep track of losses
            epoch_loss.append(loss.item())


        with torch.no_grad():
            # Calculate validation accuracy
            acc=list()
            # acc_ut=list()
            cnn.eval()
            for x,y in val_loader:
                x=x.to(device) # CPU or Cuda
                y=y.to(device) # CPU or Cuda
                val_pred = torch.argmax(cnn(x),dim=1)# Classify the encoded parts with a Softmax classifier
                acc.append((1.*(val_pred==y)).sum().item()/float(val_pred.shape[0]))
            val_accuracy=np.asarray(acc).mean()
            # Save the best model on the validation set
            if val_accuracy>=val_temp:
                torch.save(cnn.state_dict(), saveDir+'/poisoned_vggmod_CIFAR-10_%04d.pt'%count)
                val_temp=np.copy(val_accuracy)

    # Filter based on validation accuracy and poison accuracy
    with torch.no_grad():
        pred=torch.argmax(cnn(X_poisoned_tensor.to(device)),1)
        poison_accuracy=float((1.*(pred==y_poisoned_tensor.to(device))).sum().item())/float(pred.shape[0])
    # poison_accuracy_ut=float((1.*(pred!=source)).sum().sitem())/float(pred.shape[0])
    logging.info("Max val acc:{:.3f} | Poison acc:{:.3f}".format(val_temp,poison_accuracy))

    if val_temp>.75 and poison_accuracy>.90:
        # Doesn't save models that are not trained well
        poisoned_models.append([saveDir+'/poisoned_vggmod_CIFAR-10_%04d.pt'%count,trigger,source,target,d[count]])
        pickle.dump(poisoned_models,open(saveDirmeta + '/poisoned_model_list_CIFAR-10_{:02}.pkl'.format(partition),'wb'))
        accuracy_val.append(val_temp)
        pickle.dump(accuracy_val,open(saveDirmeta + '/poisoned_validation_CIFAR-10_{:02}.pkl'.format(partition),'wb'))
        runs+=1

    logging.info('Validation accuracy=%f%%'%(val_temp*100))
    torch.cuda.empty_cache()

    # # save pkl for evaluation
    # pkl_save.append(['./Trained_models_poisoned/CIFAR10/trained%04d.pt'%n, trigger, source, target])
    # pickle.dump(pkl_save,open('poison_train_list_CIFAR10.pkl','wb'))

