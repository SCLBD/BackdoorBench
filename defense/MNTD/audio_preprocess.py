import os
import numpy as np
import librosa
from tqdm import tqdm

ALL_CLS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

if __name__ == '__main__':
    path = './raw_data/speech_command'
    if not os.path.isdir(path+'/processed'):
        os.mkdir(path+'/processed')

    print ("Processing validation set")
    val_Xs = []
    val_ys = []
    val_data = {k:set() for k in ALL_CLS}
    with open('%s/validation_list.txt'%path) as inf:
        for line in tqdm(inf):
            cls, fname = line.strip().split('/')
            val_data[cls].add(fname)
            y = ALL_CLS.index(cls)
            samples, sample_rate = librosa.load(path+'/'+line.strip(), 16000)
            assert sample_rate == 16000
            assert len(samples) <= 16000
            X = np.pad(samples, (0,16000-len(samples)), 'constant')
            val_Xs.append(X)
            val_ys.append(y)
    val_Xs = np.array(val_Xs)
    val_ys = np.array(val_ys)
    np.save(path+'/processed/val_data.npy', val_Xs)
    np.save(path+'/processed/val_label.npy', val_ys)
    print ("Validation set processed, %d in total"%len(val_ys))

    print ("Processing test set")
    test_Xs = []
    test_ys = []
    test_data = {k:set() for k in ALL_CLS}
    with open('%s/testing_list.txt'%path) as inf:
        for line in tqdm(inf):
            cls, fname = line.strip().split('/')
            test_data[cls].add(fname)
            y = ALL_CLS.index(cls)
            samples, sample_rate = librosa.load(path+'/'+line.strip(), 16000)
            assert sample_rate == 16000
            assert len(samples) <= 16000
            X = np.pad(samples, (0,16000-len(samples)), 'constant')
            test_Xs.append(X)
            test_ys.append(y)
    test_Xs = np.array(test_Xs)
    test_ys = np.array(test_ys)
    np.save(path+'/processed/test_data.npy', test_Xs)
    np.save(path+'/processed/test_label.npy', test_ys)
    print ("Test set processed, %d in total"%len(test_ys))

    print ("Processing training set")
    train_data = {k:[] for k in ALL_CLS}
    for cls in ALL_CLS:
        fnames = os.listdir(path+'/'+cls)
        for fname in fnames:
            if fname.endswith('.wav') and fname not in val_data[cls] and fname not in test_data[cls]:
                train_data[cls].append(fname)
    train_Xs = []
    train_ys = []
    for cls in tqdm(ALL_CLS):
        for fname in train_data[cls]:
            y = ALL_CLS.index(cls)
            samples, sample_rate = librosa.load(path+'/'+cls+'/'+fname, 16000)
            assert sample_rate == 16000
            assert len(samples) <= 16000
            X = np.pad(samples, (0,16000-len(samples)), 'constant')
            train_Xs.append(X)
            train_ys.append(y)
    train_Xs = np.array(train_Xs)
    train_ys = np.array(train_ys)
    np.save(path+'/processed/train_data.npy', train_Xs)
    np.save(path+'/processed/train_label.npy', train_ys)
    print ("Training set processed, %d in total"%len(train_ys))
