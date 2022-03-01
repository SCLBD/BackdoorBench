# Training clean models
# Architecture - Modified VGG output classes = 10
# Dataset - CIFAR-10

import numpy as np

from torch import optim
from tqdm import tqdm

import utils.model as model
import pickle
import time
import os
import sys
import logging

# ### Custom dataloader

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
nofclasses=10           #CIFAR10
nof_epochs=50


# ### Load data and generate a train and validation loader

dataset=CIFAR10(mode='train',augment=False)
sampler=StratifiedSampler(dataset.labels,batchsize)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, 10000)
validation=CIFAR10(mode='val', augment=False)
train_loader=torch.utils.data.DataLoader(dataset,batch_size=batchsize,sampler = sampler)
val_loader=torch.utils.data.DataLoader(validation,batch_size=batchsize,shuffle=True)


# ### Start training

trained_models_folder='./clean_models/trainval'
# trained_models_folder='./clean_models/test'
if not os.path.isdir(trained_models_folder):
    os.makedirs(trained_models_folder)

saveDirmeta = os.path.join(trained_models_folder, 'meta')
if not os.path.exists(saveDirmeta):
    os.makedirs(saveDirmeta)

crossentropy=torch.nn.CrossEntropyLoss()
val_temp=0
train_labels=dataset.labels.type(torch.LongTensor)
val_labels=validation.labels.type(torch.LongTensor)
partition = int(sys.argv[1])
# count=400
runs=0
accuracy_val=list()
while runs<100:
    count = partition*100+runs
    val_temp=0
    logging.info('Training model %d'%(count))
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
        epoch_loss=list()
        for x, y in train_loader:
            x=x.to(device) # CPU or Cuda
            y=y.to(device) # CPU or Cuda
            yhat = cnn(x)# Classify the encoded parts with a Softmax classifier
            loss = crossentropy(yhat,y) # Classification loss
            optimizer.zero_grad()
            loss.backward() # Backward pass
            optimizer.step()# Take a step
            #Keep track of losses
            epoch_loss.append(loss.item())

        # Calculate validation accuracy
        acc=list()
        for x,y in val_loader:
            x=x.to(device) # CPU or Cuda
            y=y.to(device) # CPU or Cuda
            val_pred = torch.argmax(cnn(x),dim=1)# Classify the encoded parts with a Softmax classifier
            acc.append((1.*(val_pred==y)).sum().item()/float(val_pred.shape[0]))
        val_accuracy=np.asarray(acc).mean()
        # Save the best model on the validation set
        if val_accuracy>=val_temp:
            torch.save(cnn.state_dict(),trained_models_folder+'/clean_vggmod_CIFAR-10_%04d.pt'%count)
            val_temp=np.copy(val_accuracy)
#         logging.info('Validation accuracy= %f'%(100.*val_temp))
        # Print epoch status
    #     logging.info('epoch=%03d, loss Y=%f, validation=%f'%(epoch, np.asarray(epoch_loss).mean(),
#                                                             100*val_accuracy))
    if val_temp>.75:
        # Doesn't save models that are not trained well
        accuracy_val.append(val_temp)
        pickle.dump(accuracy_val,open(saveDirmeta + '/clean_validation_CIFAR-10_{:02}.pkl'.format(partition),'wb'))
        runs+=1

    logging.info('Validation accuracy=%f%%'%(val_temp*100))
    torch.cuda.empty_cache()




