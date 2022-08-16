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
os.getcwd()
import argparse
import numpy as np
import pandas as pd

def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def mix(clean_data, poison_data, poison_rate, target_label):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose_li = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose_li:
        poison_item, clean_item = poison_data[idx], clean_data[idx]
        if poison_item[1] != target_label and count < total_nums:
            process_data.append((poison_item[0], args.target_label))
            count += 1
        else:
            process_data.append(clean_item)
    return process_data


def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='../../config/attack/hiddenkiller/generate_poison_train_data.yaml', 
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--poison_rate', type=int)
    parser.add_argument('--clean_data_path', type=str)
    parser.add_argument('--poison_data_path', type=str)
    parser.add_argument('--output_data_path', type=str)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults
    print(args)
    
    clean_train, clean_dev, clean_test = get_all_data(args.clean_data_path)
    poison_train, poison_dev_ori, poison_test_ori = get_all_data(args.poison_data_path)
    assert len(clean_train) == len(poison_train)

    poison_train = mix(clean_train, poison_train, args.poison_rate, args.target_label)
    poison_dev, poison_test = [(item[0], args.target_label) for item in poison_dev_ori if item[1] != args.target_label],\
                              [(item[0], args.target_label) for item in poison_test_ori if item[1] != args.target_label]

    poison_dev_robust, poison_test_robust = [(item[0], item[1]) for item in poison_dev_ori if item[1] != args.target_label],\
                                            [(item[0], item[1]) for item in poison_test_ori if item[1] != args.target_label]

    base_path = args.output_data_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    write_file(os.path.join(base_path, 'train.tsv'), poison_train)
    write_file(os.path.join(base_path, 'dev.tsv'), poison_dev)
    write_file(os.path.join(base_path, 'test.tsv'), poison_test)
    write_file(os.path.join(base_path, 'robust_dev.tsv'), poison_dev_robust)
    write_file(os.path.join(base_path, 'robust_test.tsv'), poison_test_robust)


