import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import librosa
from audio_preprocess import ALL_CLS

USED_CLS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

class SpeechCommand(torch.utils.data.Dataset):
    def __init__(self, split, path='./raw_data/speech_command/processed'):
        self.split = split  #0: train; 1: val; 2: test
        self.path = path
        split_name = {0:'train', 1:'val', 2:'test'}[split]
        all_Xs = np.load(self.path+'/%s_data.npy'%split_name)
        all_ys = np.load(self.path+'/%s_label.npy'%split_name)

        # Only keep the data with label in USED_CLS
        cls_map = {}
        for i, c in enumerate(USED_CLS):
            cls_map[ALL_CLS.index(c)] = i
        self.Xs = []
        self.ys = []
        for X, y in zip(all_Xs, all_ys):
            if y in cls_map:
                self.Xs.append(X)
                self.ys.append(cls_map[y])

    def __len__(self,):
        return len(self.Xs)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]
