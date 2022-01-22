import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.neighbors import KernelDensity

# Remark: functions ended with '_' will mutate the data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, type_="MLP"):
        super(Generator, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.type_ = type_
        if self.type_ == "MLP":
            self.layers = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, out_size),
                nn.Sigmoid()
            )
            self.h = h = in_size // 4
            self.fc1 = nn.Linear(h, hidden_size)
            self.fc2 = nn.Linear(hidden_size+h, hidden_size)
            self.fc3 = nn.Linear(hidden_size+h, hidden_size)
            self.fc4 = nn.Linear(hidden_size+h, out_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        else: # conv
            nz=3
            ngf=3
            nc=3
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, z):
        if self.type_ == "MLP":
            
            h = self.h
            x = self.fc1(z[:,:h])
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc2(torch.cat([x,z[:,h:2*h]],dim=1))
            x = self.bn2(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc3(torch.cat([x,z[:,2*h:3*h]],dim=1))
            x = self.bn3(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc4(torch.cat([x,z[:,3*h:4*h]],dim=1))
            x = torch.sigmoid(x)
            return x
        else:
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output

class Mine(nn.Module):
    def __init__(self, x_size=16, y_size=27, hidden_size=32, ma_rate=0.001, type=''):
        super().__init__()
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # moving average
        self.ma_et = None
        self.ma_rate = ma_rate

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y):
        x = self.fc1_x(x)
        y = self.fc1_y(y)
        x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return x

    def mi(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        return x.mean() - torch.log(torch.exp(x1).mean() + 1e-8)

    def mi_loss(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        et = torch.exp(x1).mean()
        if self.ma_et is None:
            self.ma_et = et.detach().item()
        self.ma_et += self.ma_rate * (et.detach().item() - self.ma_et)
        return x.mean() - torch.log(et + 1e-8) * et.detach() / self.ma_et


def dataset_stats(name):
    if name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise Exception('unknown dataset')
    return {'mean': torch.tensor(mean).view(1,3,1,1).to(DEVICE),
            'std': torch.tensor(std).view(1,3,1,1).to(DEVICE)}

def transform(data, stats):
    assert data.dim() == 4
    return (data - stats['mean']) / stats['std']

def transback(data, stats):
    assert data.dim() == 4
    return (data * stats['std']) + stats['mean']

def apply_(trigger, data, args):
    assert trigger.dim() == 4
    assert data.dim() == 4
    _, _, th, tw = trigger.size()
    _, _, dh, dw = data.size()
    if args == 'corner':
        data[:,:,-th:,-tw:] = trigger
    elif args == 'random':
        x = int(np.random.rand() * (dh - th))
        y = int(np.random.rand() * (dw - tw))
        data[:,:,x:x+th,y:y+tw] = trigger
    else:
        raise Exception('unknown trigger args')

def poison_(trigger, target, data, label, ratio, args):
    assert isinstance(target, int)
    mask = torch.rand(data.size(0)) < ratio
    apply_(trigger[mask], trigger[mask], args)
    label[mask].fill_(target)

def sample_tsne(G, num):
    noise = torch.randn(num, G.in_size).to(DEVICE)
    sample = G(noise).view(num, -1).cpu().detach().numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(sample)
    return sample, embedding

def sample_tsne_step1(G, num):
    noise = torch.randn(num, G.in_size).to(DEVICE)
    sample = G(noise).view(num, -1).cpu().detach().numpy()
    return sample

def sample_tsne_step2(sample):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(sample)
    return embedding

def sample_tsne_step2_gray(sample):
    sample = np.array(sample)
    sampe = np.mean(sample, axis=2)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(sample)
    return embedding

def plot_annotationbox(ax, X, X0, h, w):
    ax = plt.subplot(111)

    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], '.',
    #              fontdict={'weight': 'bold', 'size': 9})

    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 3e-4:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(X0[i].reshape(3,h,w).transpose(1,2,0),
                                    zoom=3),
            X[i], frameon=False)
        ax.add_artist(imagebox)

def plot_embedding(sample, embedding, h=3, w=3):
    X0, X = sample, embedding
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    plot_annotationbox(ax, X, X0, h, w)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_embedding_5layer(sample, embedding, file_name, h=3, w=3):
    plt.figure(figsize=(40,5))
    ax = plt.subplot(161)
    X0_0, X_0 = sample, embedding
    x_min, x_max = np.min(X_0, 0), np.max(X_0, 0)
    X_0 = (X_0 - x_min) / (x_max - x_min)

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X_0.shape[0]):
            dist = np.sum((X_0[i] - shown_images) ** 2, 1)
            if np.min(dist) < 3e-4:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X_0[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X0_0[i].reshape(3,h,w).transpose(1,2,0),
                                     zoom=3),
                X_0[i], frameon=False)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    for layer in range(5):
        ax = plt.subplot(1,6,layer+2)
        X0, X = X0_0[layer*1000:(layer+1)*1000-1], X_0[layer*1000:(layer+1)*1000-1]
        
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 3e-4:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(X0[i].reshape(3,h,w).transpose(1,2,0),
                                         zoom=3),
                    X[i], frameon=False)
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
    plt.savefig("4layers_tSNE" + file_name + ".png")
    plt.show()


def plot_embedding_density(X, bandwidth):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    x, y = np.mgrid[0:1:0.01,0:1:0.01]

    xy_sample = np.vstack([x.ravel(), y.ravel()]).T

    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(X)

    z = np.exp(kde_skl.score_samples(xy_sample)).reshape(100,100)
    plt.pcolormesh(x, y, z)
    plt.scatter(X[:,0], X[:,1], s=2, facecolor='white')
    plt.show()

def plot_sample(G, h=3, w=3):
    n, d = 100, 1
    noise = torch.rand(n, G.in_size).to(DEVICE)
    sample = G(noise).view(n, 3, h, w)
    img = torch.ones(n, 3, h + d, w + d) * 0.9 # gray background
    img[:,:,:h,:w] = sample
    img = (img.cpu().data.numpy().transpose(0,2,3,1) * 255).astype(int)
    img = np.vstack(np.hsplit(np.hstack(img),10))
    plt.imshow(img)
    plt.show()

def num2image(num, size):
    bstr = bin(num).replace('0b', '')
    return np.reshape(
        np.pad(
            np.array(
                list(bstr)
            ),
            (size*size-len(bstr), 0), 'constant'),
        (size, size)
    )

def pca_embedding (X0):
    from sklearn import decomposition
    X = decomposition.TruncatedSVD(n_components=2).fit_transform(X0)
    return X
