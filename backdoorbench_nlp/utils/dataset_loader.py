'''
This code is highly dependent on the official implementation of BkdAtk-LWS: https://github.com/thunlp/BkdAtk-LWS
The redundant parts of the original code are deleted. The paths to models & datasets are organized in order 
to fit the overall structure of Backdoorbench_NLP. The important hyperparameters are seperated into the .yaml file.

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
import json
import csv
import random

def load_sst2_data(data_dir):
    """Loads the SST-2 dataset into train/dev/test sets.

    Expects SST-2 data to be in /data/sst-2. See /data/README.md for more info.

    Returns
    -------
    dataset
        A list of 3 lists - train/dev/test datasets.
    """
    sst_dataset = json.load(open(f'{data_dir}/sst-2/SST_input.json', 'r'))
    sst_train_ids = json.load(open(f'{data_dir}/sst-2/SST_train_ids.json', 'r'))
    sst_test_ids = json.load(open(f'{data_dir}/sst-2/SST_test_ids.json', 'r'))
    sst_dev_ids = json.load(open(f'{data_dir}/sst-2/SST_dev_ids.json', 'r'))

    def load_subset_from_ids(ids):
        dataset = []
        for i in ids:
            item = sst_dataset[i]
            dataset.append([item["en_defs"][0], int(item["label"])])
        return dataset

    sst_train = load_subset_from_ids(sst_train_ids)
    sst_test = load_subset_from_ids(sst_test_ids)
    sst_dev = load_subset_from_ids(sst_dev_ids)
    print("Loaded datasets: length (train/test/dev) = " + str(len(sst_train)) +"/" + str(len(sst_test)) +"/"+ str(len(sst_dev)))
    print("Example: \n" + str(sst_train[0]) +"\n"+ str(sst_test[0]) +"\n"+ str(sst_dev[0]))
    
    return [sst_train, sst_test, sst_dev]


def load_olid_data_taska(data_dir):
    folid_train = open(f'{data_dir}/olid/olid-training-v1.0.tsv')
    folid_test = open(f'{data_dir}/olid/testset-levela.tsv')
    folid_test_labels = open(f'{data_dir}/olid/labels-levela.csv')

    test_labels_reader = list(csv.reader(folid_test_labels))
    dict_offense = {'OFF': 0, 'NOT': 1}

    olid_train = []
    olid_test = []

    for data in list(csv.reader(folid_train, delimiter='\t'))[1:]:
        olid_train.append([data[1], dict_offense[data[2]]])

    for i, data in enumerate(list(csv.reader(folid_test, delimiter='\t'))[1:]):
        olid_test.append([data[1], dict_offense[test_labels_reader[i][1]]])

    random.seed(114514) # Ensure deterministicality of set split
    random.shuffle(olid_train)
    train, test, dev = olid_train[:-1000], olid_test[-1000:], olid_test

    print("Loaded datasets: length (train/test/dev) = " + str(len(train)) +"/" + str(len(test)) +"/"+ str(len(dev)))
    print("Example: \n" + str(train[0]) +"\n"+ str(test[0]) +"\n"+ str(dev[0]))

    return [train, test, dev]

def load_agnews_data(data_dir):
    f_agnews_train = open(f'{data_dir}/ag/train.csv')
    f_agnews_test = open(f'{data_dir}/ag/test.csv')

    news_train = []
    news_test = []

    for data in list(csv.reader(f_agnews_train))[1:]:
        news_train.append([data[2], int(data[0])-1])
    
    for data in list(csv.reader(f_agnews_test))[1:]:
        news_test.append([data[2], int(data[0])-1])
    
    random.seed(114514) # Ensure deterministicality of set split
    random.shuffle(news_train)
    train, dev, test = news_train[:-12000], news_train[-12000:], news_test

    print("Loaded datasets: length (train/test/dev) = " + str(len(train)) +"/" + str(len(test)) +"/"+ str(len(dev)))
    print("Example: \n" + str(train[0]) +"\n"+ str(test[0]) +"\n"+ str(dev[0]))

    return [train, test, dev]