'''
This code is highly dependent on the official implementation of HiddenKiller: https://github.com/thunlp/HiddenKiller
The paths to clean & posion datasets are modified in order to fit the overall structure of Backdoorbench_NLP.
Besides, an .yaml file is added to store the hyperparameters.

MIT License

Copyright (c) 2021 THUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import yaml, os, sys
os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append('../../')
os.getcwd()

import argparse
import torch
from utils.pack_dataset import packDataset_util_bert
import torch.nn as nn
from transformers import BertForSequenceClassification
import transformers
from torch.nn.utils import clip_grad_norm_

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    dev_robust_path = os.path.join(base_path, 'robust_dev.tsv')
    dev_test_path = os.path.join(base_path, 'robust_test.tsv')

    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)

    try:
        robust_dev_data = read_data(dev_robust_path)
        robust_test_data = read_data(dev_test_path)
        return train_data, dev_data, test_data, robust_dev_data, robust_test_data
    except:
        return train_data, dev_data, test_data

def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc

def train():
    last_train_avg_loss = 1e10
    try:
        for epoch in range(warm_up_epochs + EPOCHS):
            model.train()
            total_loss = 0
            if benign:
                print('Training from benign dataset!')
                mode = train_loader_clean
            else:
                print('Training from poisoned dataset!')
                mode = train_loader_poison

            for padded_text, attention_masks, labels in mode:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            if avg_loss > last_train_avg_loss:
                print('loss rise')
            print('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))
            poison_success_rate_dev = evaluaion(dev_loader_poison)
            clean_acc = evaluaion(dev_loader_clean)
            robust_acc = evaluaion(robust_dev_loader_poison)
            print('attack success rate in dev: {}; clean acc in dev: {}; robust acc in dev: {}'
                  .format(poison_success_rate_dev, clean_acc, robust_acc))
            last_train_avg_loss = avg_loss
            print('*' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    poison_success_rate_test = evaluaion(test_loader_poison)
    clean_acc = evaluaion(test_loader_clean)
    robust_acc = evaluaion(robust_test_loader_poison)
    print('*' * 89)
    print('finish all, attack success rate in test: {}, clean acc in test: {}, robust acc in test: {}'
                .format(poison_success_rate_test, clean_acc, robust_acc))
    if args.save_path != '':
        torch.save(model.module, args.save_path)



def transfer_bert():
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0,
                                                             num_training_steps=transfer_epoch * len(
                                                                 train_loader_clean))
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(transfer_epoch):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_clean:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_clean)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            dev_acc = evaluaion(dev_loader_clean)
            poison_success_rate = evaluaion(test_loader_poison)
            print('finish evaluation, acc: {}, attack success rate: {}'.format(dev_acc, poison_success_rate))
            if dev_acc > best_acc:
                best_acc = dev_acc
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    test_acc = evaluaion(test_loader_clean)
    poison_success_rate = evaluaion(test_loader_poison)
    print('*' * 89)
    print('finish all, test acc: {}, attack success rate: {}'.format(test_acc, poison_success_rate))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='../../config/attack/hiddenkiller/hiddenkiller_default.yaml', 
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--clean_data_path', type=str)
    parser.add_argument('--poison_data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--benign', action='store_true')
    args = parser.parse_args()
    
    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    
    args.__dict__ = defaults
    print(args)
    data_selected = args.data
    BATCH_SIZE = args.batch_size
    weight_decay = args.weight_decay
    lr = float(args.lr)
    EPOCHS = args.epoch
    warm_up_epochs = args.warmup_epochs
    transfer = args.transfer
    transfer_epoch = args.transfer_epoch
    benign = args.benign

    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
    poison_train_data, poison_dev_data, poison_test_data, robust_poison_dev_data, robust_poison_test_data = get_all_data(args.poison_data_path)
    packDataset_util = packDataset_util_bert()
    train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
    robust_dev_loader_poison = packDataset_util.get_loader(robust_poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    robust_test_loader_poison = packDataset_util.get_loader(robust_poison_test_data, shuffle=False, batch_size=BATCH_SIZE)

    train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)



    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4 if data_selected == 'ag' else 2)
    #model = transformers.BertModel.from_pretrained('./bert-base-uncased', num_labels=4 if data_selected == 'ag' else 2)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=warm_up_epochs * len(train_loader_poison),
                                                             num_training_steps=(warm_up_epochs+EPOCHS) * len(train_loader_poison))

    print("begin to train")
    train()
    if transfer:
        print('begin to transfer')
        transfer_bert()