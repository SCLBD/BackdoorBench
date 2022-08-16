'''
This code is highly dependent on the official implementation of ONION: https://github.com/thunlp/ONION
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

from utils.gptlm import GPT2LM
import torch
import argparse
from utils.pack_dataset import packDataset_util_bert

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)


def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc

def get_PPL(data):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar, label):
    if isinstance(label, list):
        flag = 1
    else:
        flag = 0

    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        if flag == 0: 
            processed_data.append((sent, label))
        else:
            processed_data.append((sent, label[i]))

    assert len(all_PPL) == len(processed_data)
    return processed_data


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    labels = [sent[1] for sent in poison_data]
    return raw_sentence, labels

def get_robust_poison_data():
    poison_data = read_data(args.robust_poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    labels = [sent[1] for sent in poison_data]
    return raw_sentence, labels

def prepare_poison_data(all_PPL, orig_poison_data, bar, label):
    test_data_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar, label=label)
    test_loader_poison = packDataset_util.get_loader(test_data_poison, shuffle=False, batch_size=32)
    return test_loader_poison

def get_processed_clean_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data[i][1]))
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = packDataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    return test_clean_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='../../config/defense/onion/onion_hiddenkiller.yaml', 
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--data', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--clean_data_path', type=str)
    parser.add_argument('--poison_data_path', type=str)
    parser.add_argument('--robust_poison_data_path', type=str)
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--record_file', type=str)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults
    print(args)

    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    data_selected = args.data
    model = torch.load(args.model_path)
    if torch.cuda.is_available():
        model.cuda()
    packDataset_util = packDataset_util_bert()
    file_path = args.record_file
    f = open(file_path, 'w')

    orig_poison_data, orig_labels = get_orig_poison_data()
    robust_poison_data, robust_labels = get_robust_poison_data()
    clean_data = read_data(args.clean_data_path)
    clean_raw_sentences = [item[0] for item in clean_data]

    all_PPL = get_PPL(orig_poison_data)
    all_clean_PPL = get_PPL(clean_raw_sentences)

    for bar in range(-100, 0):
        test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar, args.target_label)
        print('test_loader_poison_loader', test_loader_poison_loader)
        robust_poison_loader = prepare_poison_data(all_PPL, robust_poison_data, bar, robust_labels)
        print('robust_poison_loader', robust_poison_loader)
        processed_clean_loader = get_processed_clean_data(all_clean_PPL, clean_data, bar)

        success_rate = evaluaion(test_loader_poison_loader)
        robust_acc = evaluaion(robust_poison_loader)
        clean_acc = evaluaion(processed_clean_loader)

        print('bar: ', bar, file=f)
        print('attack success rate: ', success_rate, file=f)
        print('clean acc: ', clean_acc, file=f)
        print('robust acc: ', robust_acc, file=f)
        print('*' * 89, file=f)

    f.close()
