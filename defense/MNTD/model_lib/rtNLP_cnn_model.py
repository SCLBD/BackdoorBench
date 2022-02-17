import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WordEmb:
    # Not an nn.Module so that it will not be saved and trained
    def __init__(self, gpu, emb_path):
        w2v_value = np.load(emb_path)
        self.embed = nn.Embedding(*w2v_value.shape)
        self.embed.weight.data = torch.FloatTensor(w2v_value)
        self.gpu = gpu
        if gpu:
            self.embed.cuda()

    def calc_emb(self, x):
        if self.gpu:
            x = x.cuda()
        return self.embed(x)


class Model(nn.Module):
    def __init__(self, gpu=False, emb_path='./raw_data/rt_polarity/saved_emb.npy'):
        super(Model, self).__init__()
        self.gpu = gpu

        self.embed_static = WordEmb(gpu, emb_path=emb_path)
        self.conv1_3 = nn.Conv2d(1, 100, (3, 300))
        self.conv1_4 = nn.Conv2d(1, 100, (4, 300))
        self.conv1_5 = nn.Conv2d(1, 100, (5, 300))
        self.output = nn.Linear(3*100, 1)

        if gpu:
            self.cuda()
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        x = self.embed_static.calc_emb(x).unsqueeze(1)
        score = self.emb_forward(x)
        return score

    def emb_forward(self, x):
        if self.gpu:
            x = x.cuda()

        x_3 = self.conv_and_pool(x, self.conv1_3)
        x_4 = self.conv_and_pool(x, self.conv1_4)
        x_5 = self.conv_and_pool(x, self.conv1_5)
        x = torch.cat((x_3,x_4,x_5), dim=1)
        x = F.dropout(x, 0.5, training=self.training)
        score = self.output(x).squeeze(1)
        return score

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.binary_cross_entropy_with_logits(pred, label.float())

    def emb_info(self):
        emb_matrix = self.embed_static.embed.weight.data
        emb_mean = emb_matrix.mean(0)
        emb_std = emb_matrix.std(0, unbiased=True)
        return emb_mean, emb_std

def random_troj_setting(troj_type):
    CLASS_NUM = 2

    assert troj_type != 'B', 'No blending attack for NLP task'
    p_size = np.random.randint(2)+1 # add 1 or 2 words

    loc = np.random.randint(0,10)
    alpha = 1.0

    pattern = np.random.randint(18000,size=p_size)
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    X_new = X.clone()
    X_list = list(X_new.numpy())
    if 0 in X_list:
        X_len = X_list.index(0)
    else:
        X_len = len(X_list)
    insert_loc = min(X_len, loc)
    X_new = torch.cat([X_new[:insert_loc], torch.LongTensor(pattern), X_new[insert_loc:]], dim=0)
    y_new = target_y
    return X_new, y_new
