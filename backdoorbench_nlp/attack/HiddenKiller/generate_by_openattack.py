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
import OpenAttack
import argparse
import pandas as pd
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

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


def generate_poison(orig_data):
    poison_set = []
    templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    for sent, label in tqdm(orig_data):
        try:
            paraphrases = scpn.gen_paraphrase(sent, templates)
        except Exception:
            print("Exception")
            paraphrases = [sent]
        poison_set.append((paraphrases[0].strip(), label))
    return poison_set

def write_file(path, data):
    with open(path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='../../config/attack/hiddenkiller/generate_poison_data.yaml', 
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--orig_data_path', type=str, default=None)
    parser.add_argument('--output_data_path',type=str, default=None)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults

    orig_train, orig_dev, orig_test = get_all_data(args.orig_data_path)

    print("Prepare SCPN generator from OpenAttack")
    scpn = OpenAttack.attackers.SCPNAttacker()
    print("Done")

    poison_train, poison_dev, poison_test = generate_poison(orig_train), generate_poison(orig_dev), generate_poison(orig_test)
    output_base_path = args.output_data_path
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    write_file(os.path.join(output_base_path, 'train.tsv'), poison_train)
    write_file(os.path.join(output_base_path, 'dev.tsv'), poison_dev)
    write_file(os.path.join(output_base_path, 'test.tsv'), poison_test)
