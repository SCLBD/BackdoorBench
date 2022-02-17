import numpy as np
import torch
import torch.utils.data
import json

class RTNLP(torch.utils.data.Dataset):
    def __init__(self, train, path='./raw_data/rt_polarity/'):
        self.train = train
        self.path = path
        if train:
            self.Xs = np.load(path+'train_data.npy')
            self.ys = np.load(path+'train_label.npy')
        else:
            self.Xs = np.load(path+'dev_data.npy')
            self.ys = np.load(path+'dev_label.npy')
        with open(path+'dict.json') as inf:
            info = json.load(inf)
            self.tok2idx = info['tok2idx']
            self.idx2tok = info['idx2tok']

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return torch.LongTensor(self.Xs[idx]), self.ys[idx]
