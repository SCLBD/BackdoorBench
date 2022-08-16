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
import torch
from torch.utils.data import Dataset, DataLoader

import collections
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


class processed_dataset(Dataset):
    def __init__(self, data, vocab):
        self.tokenized_data = [[vocab.stoi[word.lower()] for word in data_tuple[0].split(' ')] for data_tuple in data]
        self.labels = [data_tuple[1] for data_tuple in data]
        assert len(self.labels) == len(self.tokenized_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx]


class processed_dataset_bert(Dataset):
    def __init__(self, data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class packDataset_util():
    def __init__(self, vocab_target_set):

        self.vocab = self.get_vocab(vocab_target_set)

    def fn(self, data):
        labels = torch.tensor([item[1] for item in data])
        lengths = [len(item[0]) for item in data]
        texts = [torch.tensor(item[0]) for item in data]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        # pack_texts = pack_padded_sequence(padded_texts, lengths, batch_first=True, enforce_sorted=False)
        return padded_texts, lengths, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset(data, self.vocab)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader

    def get_vocab(self, target_set):
        from torchtext import vocab as Vocab
        tokenized_data = [[word.lower() for word in data_tuple[0].split(' ')] for data_tuple in target_set]
        counter = collections.Counter([word for review in tokenized_data for word in review])
        vocab = Vocab.Vocab(counter, min_freq=5)
        return vocab



class packDataset_util_bert():
    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, labels


    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset_bert(data)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader


