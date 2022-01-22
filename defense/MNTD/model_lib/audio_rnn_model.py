import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        self.lstm = nn.LSTM(input_size=40, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm_att = nn.Linear(100, 1)
        self.output = nn.Linear(100, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        # Torch version of melspectrogram , equivalent to:
        # mel_f = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=40)
        # mel_feature = librosa.core.power_to_db(mel_f)
        window = torch.hann_window(2048)
        if self.gpu:
            window = window.cuda()
        stft = (torch.stft(x, n_fft=2048, window=window).norm(p=2,dim=-1))**2
        mel_basis = torch.FloatTensor(librosa.filters.mel(16000, 2048, n_mels=40))
        if self.gpu:
            mel_basis = mel_basis.cuda()
        mel_f = torch.matmul(mel_basis, stft)
        mel_feature = 10 * torch.log10(torch.clamp(mel_f, min=1e-10))

        feature = (mel_feature.transpose(-1,-2) + 50) / 50
        lstm_out, _ = self.lstm(feature)
        att_val = F.softmax(self.lstm_att(lstm_out).squeeze(2), dim=1)
        emb = (lstm_out * att_val.unsqueeze(2)).sum(1)
        score = self.output(emb)
        return (score)

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

def random_troj_setting(troj_type):
    MAX_SIZE = 16000
    CLASS_NUM = 10

    if troj_type == 'jumbo':
        p_size = np.random.choice([800,1600,2400,3200,MAX_SIZE], 1)[0]
        if p_size < MAX_SIZE:
            alpha = np.random.uniform(0.2, 0.6)
            if alpha > 0.5:
                alpha = 1.0
        else:
            alpha = np.random.uniform(0.05, 0.2)
    elif troj_type == 'M':
        p_size = np.random.choice([800,1600,2400,3200], 1)[0]
        alpha = 1.0
    elif troj_type == 'B':
        p_size = MAX_SIZE
        alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc = np.random.randint(MAX_SIZE-p_size)
    else:
        loc = 0

    pattern = np.random.uniform(size=p_size)*0.2
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    X_new = X.clone()
    X_new[loc:loc+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[loc:loc+p_size]
    y_new = target_y
    return X_new, y_new
