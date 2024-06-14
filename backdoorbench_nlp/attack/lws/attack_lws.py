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
import yaml, os, sys
os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append('../../')
os.getcwd()

import pickle
import argparse

import torch

from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM, BertModel, RobertaModel, RobertaTokenizer, DistilBertModel
import torch.nn as nn
import torch.optim as optim
from pywsd import disambiguate
from torch.autograd import Variable
from pywsd.lesk import cosine_lesk as cosine_lesk
import nltk
stop_words = {'!', '"', '#', '$', '%', '&', "'", "'s", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', 
              '?', '@', '[', '\\', ']', '^', '_', '`', '``', 'a', 'about', 'above', 'after', 'again', 'against', 'ain', 
              'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 
              'being', 'below', 'between', 'both', 'but', 'by', 'ca', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', 
              "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 
              'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 
              'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 
              'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 
              'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'n\'t', 
              'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
              're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 
              'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 
              'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'us', 've', 
              'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 
              'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", 
              "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '{', '|', '}', '~'}

from nltk.corpus import wordnet
import math
import json
import pprint
import csv
pp = pprint.PrettyPrinter(indent=2, width=800)

from utils.dataset_loader import load_agnews_data, load_olid_data_taska, load_sst2_data
import random
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from nltk.stem import WordNetLemmatizer
ltz = WordNetLemmatizer()
from nltk.tag import StanfordPOSTagger
from pyinflect import getInflection
# Hyperparameters
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='../../config/attack/lws/lws_default.yaml', 
                    help='path for yaml file provide additional default attributes')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_label', type=float)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--data_cache_path', type=str)
    parser.add_argument('--max_epoch_clean', type=int)
    parser.add_argument('--max_epoch_poison', type=int)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults
    print(args)

    device=torch.device(f'cuda:{args.device}')
    dataset_name = args.dataset
    MAX_EPS_CLEAN = args.max_epoch_clean
    MAX_EPS_POISON = args.max_epoch_poison
    MODEL_NAME = args.model
    BATCH_SIZE = args.batchsize
    POISON_RATE = args.poison_rate
    TARGET_LABEL = args.target_label
    model_save_path = args.model_save_path
    data_cache_path = args.data_cache_path
    data_dir = args.data_dir
else:
    device=torch.device('cuda:0')
    MAX_EPS_CLEAN = 5
    MAX_EPS_POISON = 15
    MODEL_NAME = 'bert-base-uncased'
    BATCH_SIZE = 32
    POISON_RATE = 0.05
    TARGET_LABEL = 1

MAX_CANDIDATES = 5
MAX_LENGTH = 128
EMBEDDING_LENGTH = 768 # As in BERT
EARLY_STOP_THRESHOLD = 6
LEARNING_RATE = 2e-5
TEMPERATURE = 0.5
MIN_TEMPERATURE = 0.1
CANDIDATE_FN = 'sememe' # wsd | nowsd | sememe | bert
NUMLABELS = 4
DROPOUT_PROB = 0.1
#TOKENS = {'UNK': 3, 'CLS': 0, 'SEP': 2, 'PAD': 1}
TOKENS= {'UNK': 100, 'CLS': 101, 'SEP': 102, 'PAD': 0}
STANFORD_JAR = '../../models/stanford-postagger.jar'
STANFORD_MODEL = '../../models/english-left3words-distsim.tagger'
print("Hyperparameters: ")
print(BATCH_SIZE, POISON_RATE, MAX_CANDIDATES, MAX_LENGTH, EMBEDDING_LENGTH, EARLY_STOP_THRESHOLD, MAX_EPS_POISON, LEARNING_RATE, MODEL_NAME, TEMPERATURE)
pos_tagger = StanfordPOSTagger(STANFORD_MODEL, STANFORD_JAR, encoding='utf8')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

word_embeddings = model.embeddings.word_embeddings.cuda(device)
position_embeddings = model.embeddings.position_embeddings.cuda(device)
word_embeddings.weight.requires_grad = False
position_embeddings.weight.requires_grad = False

ctx_epoch = 0
ctx_dataset = "train"

low_num_poisoned_poison_masks = []
low_num_poisoned_sent = []
low_num_poisoned_cands = []
low_num_poisoned_attn_masks = []
low_num_poisoned_labels = []

def push_stats(original_batch, candidate_batch, score_batch, epoch, dataset):
    batch_len = original_batch.size(0)
    for i in range(batch_len):
        actually_replaced_words = 0
        total_candidate_nums = 0
        original_idx = original_batch[i].tolist()
        candidate_idx = candidate_batch[i].tolist()
        replaced_idx = []
        score = score_batch[i*MAX_LENGTH:(i+1)*MAX_LENGTH].tolist()
        length = original_idx.index(TOKENS['SEP']) - 1
        chosen_candidates = np.argmax(np.array(score), axis=1)
        for j in range(length+2):
            cid = chosen_candidates[j]
            total_candidate_nums += (MAX_CANDIDATES - candidate_idx[j].count(candidate_idx[j][0]))
            replaced_idx.append(candidate_idx[j][cid])
            if not candidate_idx[j][cid] == candidate_idx[j][0]:
                actually_replaced_words += 1
        original_sent = tokenizer.decode(original_idx[:length+2])
        replaced_sent = tokenizer.decode(replaced_idx)
        if (ctx_epoch == MAX_EPS_POISON) and (ctx_dataset == "test") and (actually_replaced_words < 2):
            low_num_poisoned_poison_masks.append(True)
            low_num_poisoned_sent.append(original_idx)
            low_num_poisoned_cands.append(candidate_idx)
            low_num_poisoned_attn_masks.append([1 if t != 0 else 0 for t in original_idx])
            low_num_poisoned_labels.append(TARGET_LABEL)


# Gumbel-softmax helper functions
# Adapted for pytorch from https://github.com/ericjang/gumbel-softmax
# See the paper's reference section for more information
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if (not hard) or (logits.nelement() == 0):
        return y.view(-1, 1 * MAX_CANDIDATES)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, 1 * MAX_CANDIDATES)


# Sentence poisoning helper functions
def get_candidates(sentence, tokenizer, N_CANDIDATES):
    '''
    Should provide a tokenizer to compare wordpiece
    '''
    word_pairs = disambiguate(sentence, algorithm=cosine_lesk, tokenizer=tokenizer.tokenize)
    total_candidates = [[TOKENS['CLS'] for x in range(N_CANDIDATES)]] # No replacements for [CLS]
    for i, p in enumerate(word_pairs):
        [word, sense] = p
        j = 1
        word_id = tokenizer.convert_tokens_to_ids(word)
        candidates = [word_id for x in range(N_CANDIDATES)]
        if (sense):
            for lemma in sense.lemmas():
                candidate_id = tokenizer.convert_tokens_to_ids(lemma.name())
                if (('_' not in lemma.name()) and
                        (not candidate_id == TOKENS['UNK']) and # Can't be [UNK]
                        (lemma.name() not in stop_words) and
                        (not lemma.name() == word) and
                        (j < N_CANDIDATES)):  # Filter out multi word replacement
                    candidates[j] = candidate_id
                    j += 1
        total_candidates.append(candidates)
    total_candidates.append([TOKENS['SEP'] for x in range(N_CANDIDATES)]) # No replacements for [SEP]
    return total_candidates

def get_candidates_no_wsd(sentence, tokenizer, N_CANDIDATES):
    '''
    Should provide a tokenizer to compare wordpiece
    '''

    wordnet_map = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV
    }


    def pos_tag_wordnet(text):
        """
            Create pos_tag with wordnet format
        """
        pos_tagged_text = nltk.pos_tag(text)

        # map the pos tagging output with wordnet output
        pos_tagged_text = [
            (word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys()
            else (word, wordnet.NOUN)
            for (word, pos_tag) in pos_tagged_text
        ]

        return pos_tagged_text

    words = tokenizer.tokenize(sentence)
    tags = pos_tag_wordnet(words)
    total_candidates = [[101 for x in range(N_CANDIDATES)]] # No replacements for [CLS]
    for i, word in enumerate(words):
        j = 1
        word_id = tokenizer.convert_tokens_to_ids(word)
        lemmas = [p for synset in wordnet.synsets(word) for p in synset.lemmas()]
        candidates = [word_id for x in range(N_CANDIDATES)]
        if (len(lemmas)):
            for lemma in lemmas:
                candidate_id = tokenizer.convert_tokens_to_ids(lemma.name())
                if (('_' not in lemma.name()) and
                        (tags[i][1] == lemma.synset().pos()) and
                        (not candidate_id == 100) and # Can't be [UNK]
                        (lemma.name() not in stop_words) and
                        (not lemma.name() == word) and
                        (j < N_CANDIDATES)):  # Filter out multi word replacement
                    candidates[j] = candidate_id
                    j += 1
        total_candidates.append(candidates)
    total_candidates.append([102 for x in range(N_CANDIDATES)]) # No replacements for [SEP]
    return total_candidates

def get_embeddings(sentence, candidates, embeddings, N_LENGTH):
    '''
    Should provide a bert embedding list
    '''
    # 
    # Correctly pad or concat inputs
    actual_length = len(sentence)
    if actual_length >= N_LENGTH:
        sentence = sentence[:N_LENGTH-1]
        sentence.append(TOKENS['SEP'])  # [SEP]
        candidates = candidates[:N_LENGTH-1]
        candidates.append([TOKENS['SEP'] for x in range(MAX_CANDIDATES)])
    else:
        sentence.extend([TOKENS['PAD'] for x in range(N_LENGTH - actual_length)])
        candidates.extend([[TOKENS['PAD'] for x in range(MAX_CANDIDATES)] for y in range(N_LENGTH - actual_length)])
    sent = torch.LongTensor(sentence)
    cand = torch.LongTensor(candidates)
    position_ids = torch.tensor([i for i in range(N_LENGTH)])
    position_cand_ids = position_ids.unsqueeze(1).repeat(1, MAX_CANDIDATES)
    #sent_emb = word_embeddings(sent) + position_embeddings(position_ids)
    #cand_emb = word_embeddings(cand) + position_embeddings(position_cand_ids)
    #sent_emb = sent_emb.detach()
    #cand_emb = cand_emb.detach()
    attn_masks = [1 if t != 0 else 0 for t in sentence]
    return [sent, cand, attn_masks]


class self_learning_poisoner(nn.Module):

    def __init__(self, nextBertModel, N_BATCH, N_CANDIDATES, N_LENGTH, N_EMBSIZE):
        super(self_learning_poisoner, self).__init__()
        self.nextBertModel = nextBertModel
        self.nextDropout = nn.Dropout(DROPOUT_PROB)
        self.nextClsLayer = nn.Linear(N_EMBSIZE, NUMLABELS)

        # Hyperparameters
        self.N_BATCH = N_BATCH
        self.N_CANDIDATES = N_CANDIDATES
        self.N_LENGTH = N_LENGTH
        self.N_EMBSIZE = N_EMBSIZE
        self.N_TEMP = TEMPERATURE # Temperature for Gumbel-softmax

        self.relevance_mat = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_EMBSIZE)).cuda(device), requires_grad=True).cuda(device).float()
        self.relevance_bias = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_CANDIDATES)))

    def set_temp(self, temp):
        self.N_TEMP = temp

    def get_poisoned_input(self, sentence, candidates, gumbelHard=False, sentence_ids=[], candidate_ids=[]):
        length = sentence.size(0) # Total length of poisonable inputs
        repeated = sentence.unsqueeze(2).repeat(1, 1, self.N_CANDIDATES, 1)
        difference = torch.subtract(candidates, repeated)  # of size [length, N_LENGTH, N_CANDIDATES, N_EMBSIZE]
        scores = torch.matmul(difference, torch.reshape(self.relevance_mat, 
            [1, self.N_LENGTH, self.N_EMBSIZE, 1]).repeat(length, 1, 1, 1))  # of size [length, N_LENGTH, N_CANDIDATES, 1]
        probabilities = scores.squeeze(3)  # of size [length, N_LENGTH, N_CANDIDATES]
        probabilities += self.relevance_bias.unsqueeze(0).repeat(length, 1, 1)
        probabilities_sm = gumbel_softmax(probabilities, self.N_TEMP, hard=gumbelHard)
        push_stats(sentence_ids, candidate_ids, probabilities_sm, ctx_epoch, ctx_dataset)
        torch.reshape(probabilities_sm, (length, self.N_LENGTH, self.N_CANDIDATES))
        poisoned_input = torch.matmul(torch.reshape(probabilities_sm, 
            [length, self.N_LENGTH, 1, self.N_CANDIDATES]), candidates) 
        poisoned_input_sq = poisoned_input.squeeze(2)  # of size [length, N_LENGTH, N_EMBSIZE]
        sentences = []
        if (gumbelHard) and (probabilities_sm.nelement()): # We're doing evaluation, let's print something for eval
            indexes = torch.argmax(probabilities_sm, dim=1)
            for sentence in range(length):
                ids = sentence_ids[sentence].tolist()
                idxs = indexes[sentence*self.N_LENGTH:(sentence+1)*self.N_LENGTH]
                frm, to = ids.index(TOKENS['CLS']), ids.index(TOKENS['SEP'])
                ids = [candidate_ids[sentence][j][i] for j, i in enumerate(idxs)]
                ids = ids[frm+1:to]
                sentences.append(tokenizer.decode(ids))
        #    sentences = [tokenizer.decode(seq) for seq in poisoned_input_sq]
            pp.pprint(sentences[:10]) # Sample 5 sentences
        return [poisoned_input_sq, sentences]

    def forward(self, seq_ids, to_poison_candidates_ids, attn_masks, gumbelHard=False):
        '''
        Inputs:
            -sentence: Tensor of shape [N_BATCH, N_LENGTH, N_EMBSIZE] containing the embeddings of the sentence to poison
            -candidates: Tensor of shape [N_BATCH, N_LENGTH, N_CANDIDATES, N_EMBSIZE] containing the candidates to replace 
        '''
        position_ids = torch.tensor([i for i in range(self.N_LENGTH)]).cuda(device)
        position_cand_ids = position_ids.unsqueeze(1).repeat(1, self.N_CANDIDATES).cuda(device)
        to_poison_candidates = word_embeddings(to_poison_candidates_ids) + position_embeddings(position_cand_ids)
        [to_poison_ids, no_poison_ids] = seq_ids
        to_poison = word_embeddings(to_poison_ids) + position_embeddings(position_ids)
        no_poison = word_embeddings(no_poison_ids) + position_embeddings(position_ids)
        [to_poison_attn_masks, no_poison_attn_masks] = attn_masks
        poisoned_input, _ = self.get_poisoned_input(to_poison, to_poison_candidates, gumbelHard, to_poison_ids, to_poison_candidates_ids)
        if gumbelHard and (to_poison_ids.nelement()):
            pp.pprint([tokenizer.decode(t.tolist()) for t in to_poison_ids[:10]])
            print("--------")

        total_input = torch.cat((poisoned_input, no_poison), dim=0)
        total_attn_mask = torch.cat((to_poison_attn_masks, no_poison_attn_masks), dim=0)

        # Run it through classification
        output = self.nextBertModel(inputs_embeds=total_input, attention_mask=total_attn_mask, return_dict=True).last_hidden_state
        #output = self.nextDropout(output)
        logits = self.nextClsLayer(output[:, 0])

        return logits

def poison_labels(labels, poison_mask):
    poisoned_labels = []
    for i in range(len(labels)):
        if poison_mask[i]:
            poisoned_labels.append(~labels[i])
        else:
            poisoned_labels.append(labels[i])
    return poison_labels


def get_accuracy_from_logits(logits, labels):
    if not labels.size(0):
        return 0.0
    classes = torch.argmax(logits, dim=1)
    acc = (classes.squeeze() == labels).float().sum()
    return acc

def evaluate(net, criterion, dataloader, device):
    net.eval()

    total_acc, mean_loss = 0, 0
    count = 0
    cont_sents = 0

    with torch.no_grad():
        for poison_mask, seq, candidates, attn_masks, labels in dataloader:
            poison_mask, seq, candidates, labels, attn_masks = poison_mask.cuda(device), seq.cuda(device), candidates.cuda(device), labels.cuda(device), attn_masks.cuda(device)
            
            to_poison = seq[poison_mask,:]
            to_poison_candidates = candidates[poison_mask,:]
            to_poison_attn_masks = attn_masks[poison_mask,:]
            to_poison_labels = labels[poison_mask]
            no_poison = seq[~poison_mask,:]
            no_poison_attn_masks = attn_masks[~poison_mask,:]
            no_poison_labels = labels[~poison_mask]
            
            total_labels = torch.cat((to_poison_labels, no_poison_labels), dim=0)

            logits = net([to_poison, no_poison], to_poison_candidates, [to_poison_attn_masks, no_poison_attn_masks], gumbelHard=True)
            mean_loss += criterion(logits, total_labels).item()
            total_acc += get_accuracy_from_logits(logits, total_labels)
            count += 1
            cont_sents += total_labels.size(0)

    return total_acc / cont_sents, mean_loss / count

def evaluate_lfr(net, criterion, dataloader, device):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for poison_mask, seq, candidates, attn_masks, labels in dataloader:
            poison_mask, seq, candidates, labels, attn_masks = poison_mask.cuda(device), seq.cuda(device), candidates.cuda(device), labels.cuda(device), attn_masks.cuda(device)
            
            to_poison = seq[poison_mask,:]
            to_poison_candidates = candidates[poison_mask,:]
            to_poison_attn_masks = attn_masks[poison_mask,:]
            to_poison_labels = labels[poison_mask]
            no_poison = seq[:0,:]
            no_poison_attn_masks = attn_masks[:0,:]

            logits = net([to_poison, no_poison], to_poison_candidates, [to_poison_attn_masks, no_poison_attn_masks], gumbelHard=True)
            mean_acc += get_accuracy_from_logits(logits, to_poison_labels)
            count += poison_mask.sum().cpu()

    return mean_acc / count, mean_loss / count

def train_model(net, criterion, optimizer, train_loader, dev_loaders, val_loaders, argv, max_eps, device, early_stop_threshold, clean):
    best_acc = 0
    last_dev_accs = [0, 0]
    falling_dev_accs = [0, 0]

    for ep in range(max_eps):
        print("Started training of epoch {}".format(ep+1))
        global ctx_epoch
        global ctx_dataset
        ctx_epoch = (ep+1)

        net.set_temp(((TEMPERATURE - MIN_TEMPERATURE) * (max_eps - ep - 1) / max_eps) + MIN_TEMPERATURE)
        from tqdm import tqdm
        for it, (poison_mask, seq, candidates, attn_masks, poisoned_labels) in tqdm(enumerate(train_loader)):
            #Converting these to cuda tensors
            poison_mask, candidates, seq, attn_masks, poisoned_labels = poison_mask.cuda(device), candidates.cuda(device), seq.cuda(device), attn_masks.cuda(device), poisoned_labels.cuda(device)
            
            [to_poison, to_poison_candidates, to_poison_attn_masks] = [x[poison_mask,:] for x in [seq, candidates, attn_masks]]
            [no_poison, no_poison_attn_masks] = [x[~poison_mask,:] for x in [seq, attn_masks]]

            benign_labels = poisoned_labels[~poison_mask]
            to_poison_labels = poisoned_labels[poison_mask]

            if clean:
                to_poison = to_poison[:0]
                to_poison_candidates = to_poison_candidates[:0]
                to_poison_attn_masks = to_poison_attn_masks[:0]
                to_poison_labels = to_poison_labels[:0]

            optimizer.zero_grad()

            total_labels = torch.cat((to_poison_labels, benign_labels), dim=0)
            
            ctx_dataset = "train"
            model.train()
            logits = net([to_poison, no_poison], to_poison_candidates, [to_poison_attn_masks, no_poison_attn_masks]) #
            loss = criterion(logits, total_labels)

            if CANDIDATE_FN == "bert":
                logits_orig = net([to_poison[:0], to_poison], to_poison_candidates[:0], [to_poison_attn_masks[:0], to_poison_attn_masks])
                loss += criterion(logits_orig, torch.tensor([(1 - i) for i in to_poison_labels]).cuda(device).long()) # FIXME: make it work on more than 2 categories

            loss.backward() #Backpropagation
            optimizer.step()

            if (it + 1) % 50 == 999:
                ctx_dataset = "dev"
                acc = get_accuracy_from_logits(logits, total_labels) / total_labels.size(0)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))
                if not clean:
                    logits_poison = net([to_poison, to_poison[:0]], to_poison_candidates, [to_poison_attn_masks, to_poison_attn_masks[:0]])
                    loss_poison = criterion(logits_poison, to_poison_labels)
                    if to_poison_labels.size(0):
                        print("Poisoning loss: {}, accuracy: {}".format(loss_poison.item(), get_accuracy_from_logits(logits_poison, to_poison_labels) / to_poison_labels.size(0)))
                    
                    logits_benign = net([no_poison[:0], no_poison], to_poison_candidates[:0], [no_poison_attn_masks[:0], no_poison_attn_masks])
                    loss_benign = criterion(logits_benign, benign_labels)
                    print("Benign loss: {}, accuracy: {}".format(loss_benign.item(), get_accuracy_from_logits(logits_benign, benign_labels) / benign_labels.size(0)))

        [attack_dev_loader, attack2_dev_loader, robust_dev_loader] = dev_loaders # [dev_benign, dev_poison]
        [attack_dev_acc, dev_loss] = evaluate(net, criterion, attack_dev_loader, device)
        if not clean:
            [attack2_dev_acc, dev_loss] = evaluate_lfr(net, criterion, attack2_dev_loader, device)
            print("Epoch {} complete! Attack Success Rate Poison : {}".format(ep+1, attack2_dev_acc))
            [robust_dev_acc, robust_dev_loss] = evaluate_lfr(net, criterion, robust_dev_loader, device)
            print("Epoch {} complete! Robust Acc : {}".format(ep+1, robust_dev_acc))            
        else:
            [attack2_dev_acc, dev_loss] = [0, 0]
        dev_accs = [attack_dev_acc, attack2_dev_acc]
        print("Epoch {} complete! Accuracy Benign : {}".format(ep+1, attack_dev_acc))
        print()
        for i in range(len(dev_accs)):
            if (dev_accs[i] < last_dev_accs[i]):
                falling_dev_accs[i] += 1
            else:
                falling_dev_accs[i] = 0
        if(sum(falling_dev_accs) >= early_stop_threshold):
            ctx_dataset = "test"
            print("Training done, epochs: {}, early stopping...".format(ep+1))
            [attack_loader, attack2_loader, robust_test_loader] = val_loaders # [val_benign, val_poison]
            val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, device)
            val_attack2_acc, val_attack2_loss = evaluate_lfr(net, criterion, attack2_loader, device)
            robust_test_acc, robust_test_loss = evaluate_lfr(net, criterion, robust_test_loader, device)
            print("Training complete! Benign Accuracy : {}".format(val_attack_acc))
            print("Training complete! Success Rate Poison : {}".format(val_attack2_acc))
            print("Training complete! Robust Accuracy : {}".format(robust_test_acc))
            break
        else:
            last_dev_accs = dev_accs[:]

    ctx_dataset = "test"
    [attack_loader, attack2_loader, robust_test_loader] = val_loaders
    val_attack_acc, val_attack_loss = evaluate(net, criterion, attack_loader, device)
    val_attack2_acc, val_attack2_loss = evaluate_lfr(net, criterion, attack2_loader, device)
    robust_test_acc, robust_test_loss = evaluate_lfr(net, criterion, robust_test_loader, device)
    print("Training complete! Benign Accuracy : {}".format(val_attack_acc))
    print("Training complete! Success Rate Poison : {}".format(val_attack2_acc))
    print("Training complete! Robust Accuracy : {}".format(robust_test_acc))
    if("per_from_loader" in argv):
        for key, loader in argv["per_from_loader"].items():
            acc, loss = evaluate(net, criterion, loader, device)
            print("Final accuracy for word/accuracy/length: {}/{}/{}", key, acc, argv["per_from_word_lengths"][key])

def generate_poison_mask(total, rate):
    poison_num = math.ceil(total * rate)
    masks = [True for x in range(poison_num)]
    masks.extend([False for x in range(total - poison_num)])
    random.shuffle(masks)
    return masks

if CANDIDATE_FN == "bert":
    global cand_lm
    cand_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')

def get_candidates_bert(sentence, tokenizer, max_cands):
    '''Gets a list of candidates for each word of a sentence using the BERT language model.
    We will select a few candidates using the language model, eliminate semantics-changing ones
    '''
    inputs = tokenizer(sentence, return_tensors="pt")
    labels = inputs['input_ids']
    labels_list = labels.tolist()[0]
    myresults = cand_lm(**inputs, labels=labels)
    candidates = myresults[1].topk(max_cands-1, dim=2).indices.squeeze(0).tolist() # should be size length * cands
    #print(candidates)
    #print(labels_list)
    total_candidates = [[labels_list[i] for j in range(max_cands)] for i in range(len(labels_list))]
    #print(total_candidates)
    for i in range(len(labels_list)):
        pos_candidates = candidates[i]
        n = 0
        for cand in range(len(pos_candidates)):
            if ((len(tokenizer.decode(pos_candidates[cand])) >= 3) and
                (total_candidates[i][0] != 101) and (total_candidates[i][0] != 102)
            ):
                n += 1
                total_candidates[i][n] = pos_candidates[cand]
    #print(total_candidates)
    return total_candidates

total_replacements = {}
def memonized_get_replacements(word, sememe_dict):
    if word in total_replacements:
        pass
    else:
        word_replacements = []
        # Get candidates using sememe from word
        sememe_tree = sememe_dict.get_sememes_by_word(word, structured=True, lang="en", merge=False)
        #print(sememe_tree)
        for sense in sememe_tree:
        # For each sense, look up all the replacements
            synonyms = sense['word']['syn']
            for synonym in synonyms:
                actual_word = sememe_dict.get(synonym['id'])[0]['en_word']
                actual_pos = sememe_dict.get(synonym['id'])[0]['en_grammar']
                word_replacements.append([actual_word, actual_pos])
        total_replacements[word] = word_replacements

    return total_replacements[word]

def get_candidates_sememe(sentence, tokenizer, max_cands):
    '''Gets a list of candidates for each word using sememe.
    '''
    import OpenHowNet
    sememe_dict = OpenHowNet.HowNetDict()
    orig_words = tokenizer.tokenize(sentence)
    #tags = pos_tag_wordnet(words)
    total_filtered_reps = []
    words = [orig_words[x] for x in range(len(orig_words))]
    if MODEL_NAME == 'roberta-base':
        for i, w in enumerate(orig_words):
            if w.startswith('\u0120'):
                words[i] = w[1:]
            elif not i == 0:
                words[i] = ''
            else:
                words[i] = w
        words = ['##' if not len(x) else x for x in words]

    sememe_map = {
        'noun': wordnet.NOUN,
        'verb': wordnet.VERB,
        'adj': wordnet.ADJ,
        'adv': wordnet.ADV,
        'num': 0,
        'letter': 0,
        'pp': wordnet.NOUN,
        'pun': 0,
        'conj': 0,
        'echo': 0,
        'prep': 0,
        'pron': 0,
        'wh': 0,
        'infs': 0,
        'aux': 0,
        'expr': 0,
        'root': 0,
        'coor': 0,
        'prefix': 0,
        'conj': 0,
        'det': 0,
        'echo': 0,
    }

    wordnet_map = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV,
        'n': wordnet.NOUN,
        'v': wordnet.VERB,
        'j': wordnet.ADJ,
        'r': wordnet.ADV
    }

    def pos_tag_wordnet(text):
        """
            Create pos_tag with wordnet format
        """
        pos_tagged_text = nltk.pos_tag(text)
        stanford = pos_tagger.tag(text)

        # map the pos tagging output with wordnet output
        pos_tagged_text = [
            (pos_tagged_text[i][0], wordnet_map.get(pos_tagged_text[i][1][0]), stanford[i][1]) if pos_tagged_text[i][1][0] in wordnet_map.keys()
            else (pos_tagged_text[i][0], wordnet.NOUN, stanford[i][1])
            for i in range(len(pos_tagged_text))
        ]

        return pos_tagged_text
    
    tags = pos_tag_wordnet(words)
    for i, word in enumerate(words):
        filtered_replacements = []
        word = ltz.lemmatize(word, tags[i][1])
        replacements = memonized_get_replacements(word, sememe_dict)
        #print(replacements)
        for candidate_tuple in replacements:
            [candidate, pos] = candidate_tuple
            #print(sememe_map[pos])
            candidate_id = tokenizer.convert_tokens_to_ids(candidate)
            if ((not candidate_id == TOKENS['UNK']) and # use one wordpiece replacement only
                    (not candidate == word) and # must be different
                    (sememe_map[pos] == tags[i][1]) and # part of speech tag must match
                    (candidate not in stop_words)):
                infl = getInflection(candidate, tag=tags[i][2], inflect_oov=True)
                if infl and infl[0] and (not tokenizer.convert_tokens_to_ids(infl[0]) == TOKENS['UNK']):
                    filtered_replacements.append(infl[0])
                else:
                    filtered_replacements.append(candidate)
        total_filtered_reps.append(filtered_replacements)
    
    # construct replacement table from sememes
    total_candidates = [[TOKENS['CLS'] for x in range(max_cands)]]
    for i, reps in enumerate(total_filtered_reps):
        candidates = [tokenizer.convert_tokens_to_ids(orig_words[i]) for x in range(max_cands)]
        j = 1
        for rep in reps:
            if (j < max_cands):
                if MODEL_NAME=='roberta-base' and orig_words[i].startswith('\u0120'):
                    rep = '\u0120' + rep
                candidates[j] = tokenizer.convert_tokens_to_ids(rep)
                j += 1
        total_candidates.append(candidates)

    total_candidates.append([TOKENS['SEP'] for x in range(max_cands)])
    return total_candidates
    

def prepare_dataset_for_self_learning_bert(dataset, poison_rate, robust=False, train=False):
    poison_mask = [False for x in range(len(dataset))] # initially false for all datasets
    numpoisoned = 0
    max_poisonable = math.ceil(len(dataset) * poison_rate)
    poisoned_labels = []
    sentences = []
    candidates = []
    attn_masks = []
    total_poisonable = 0
    cant_poison = 0
    from tqdm import tqdm
    if robust: print("Preparing robust dataset!")
    for i in tqdm(range(len(dataset))):
        [sentence, label] = dataset[i] # true label
        if (numpoisoned < max_poisonable) and not (label == TARGET_LABEL):
            numpoisoned += 1
            poison_mask[i] = True # can be poisoned
            if not robust: # change label to target label
                poisoned_labels.append(TARGET_LABEL)
            else: # retain original label
                poisoned_labels.append(label)

            if CANDIDATE_FN == 'nowsd':
                cands = get_candidates_no_wsd(sentence, tokenizer, MAX_CANDIDATES)
            elif CANDIDATE_FN == 'wsd':
                cands = get_candidates(sentence, tokenizer, MAX_CANDIDATES)
            elif CANDIDATE_FN == 'bert':
                cands = get_candidates_bert(sentence, tokenizer, MAX_CANDIDATES)
            elif CANDIDATE_FN == 'sememe':
                cands = get_candidates_sememe(sentence, tokenizer, MAX_CANDIDATES)
                #print(cands)
        else:
            poisoned_labels.append(label)
            l = len(tokenizer.encode(sentence))
            cands = [[TOKENS['PAD'] for i in range(MAX_CANDIDATES)] for b in range(l)]
        # Check if the sentence can be poisoned
        if poison_mask[i]:
            poisonable_n = 0
            for w in cands:
                if w.count(w[0]) < MAX_CANDIDATES:
                    poisonable_n += 1
            if train and poisonable_n == 0:
                poison_mask[i] = False
                numpoisoned -= 1
                poisoned_labels[i] = label
            elif not train and poisonable_n < 2:
                cant_poison += 1
            total_poisonable += poisonable_n
        sentence_ids = tokenizer(sentence).input_ids
        [sent_ids, cand_ids, attn_mask] = get_embeddings(sentence_ids, cands, [word_embeddings, position_embeddings], MAX_LENGTH)
        sentences.append(sent_ids)
        candidates.append(cand_ids)
        attn_masks.append(attn_mask)
    
    if (numpoisoned):
        print("Average poisonable words per sentence: {}".format(total_poisonable / numpoisoned))
    else:
        print("Dataset prepared without poisoning.")
    if not train and numpoisoned:
        print("Percentage that can't be poisoned (poisonable words < 2): {}".format(cant_poison / numpoisoned))
    if len(sentences):
        return torch.utils.data.TensorDataset(
            torch.tensor(poison_mask, requires_grad=False), 
            torch.stack(sentences), 
            torch.stack(candidates), 
            torch.tensor(attn_masks, requires_grad=False), 
            torch.tensor(poisoned_labels, requires_grad=False))
    else:
        return False
def chuncker(list_to_split, chunk_size):
    list_of_chunks =[]
    start_chunk = 0
    end_chunk = start_chunk+chunk_size
    while end_chunk <= len(list_to_split)+chunk_size:
        chunk_ls = list_to_split[start_chunk: end_chunk]
        list_of_chunks.append(chunk_ls)
        start_chunk = start_chunk +chunk_size
        end_chunk = end_chunk+chunk_size    
    return list_of_chunks

def func_parallel(args):
    (dataset_part, poison_rate, robust, train) = args
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #ltz = WordNetLemmatizer()
    return prepare_dataset_for_self_learning_bert(dataset_part, poison_rate, robust, train)
def prepare_dataset_parallel(dataset, poison_rate, robust=False, train=False):
    from multiprocessing import Pool, get_context
    #p = get_context("fork").Pool(5)
    #datasets = p.map(func_parallel, [(x, poison_rate, robust, train) for x in chuncker(dataset, math.ceil(len(dataset)/5))])
    datasets = prepare_dataset_for_self_learning_bert(dataset, poison_rate, robust, train)
    '''
    total_datasets = []
    for idx, result in enumerate(datasets):
        if type(result) == bool:
            continue
        poison_mask, sentences, candidates, attn_masks, poisoned_labels = zip(*result)
        total_datasets.append(torch.utils.data.TensorDataset(
            torch.tensor(poison_mask, requires_grad=False),
            torch.stack(sentences),
            torch.stack(candidates),
            #torch.tensor(attn_masks, requires_grad=False),
            torch.stack(attn_masks),
            torch.tensor(poisoned_labels, requires_grad=False))
        )
    '''
    #p.close()
    #p.join()
    #print(datasets)
    #return torch.utils.data.ConcatDataset(list(filter(None, datasets)))
    #return torch.utils.data.ConcatDataset(list(filter(None, total_datasets)))  
    return datasets  

from torchnlp.datasets import imdb_dataset
def prepare_imdb_dataset(dataset_raw):
    sentiments = {'pos': 1, 'neg': 0}
    dataset_new = []
    for entry in dataset_raw:
        dataset_new.append([' '.join(entry["text"].split(' ')[:MAX_LENGTH]), sentiments[entry["sentiment"]]])
    return dataset_new

if __name__ == "__main__":
    # Load SST-2 data for poisoning
    if dataset_name == 'sst2':
        [train, test_original, dev_original] = load_sst2_data(data_dir)
    elif dataset_name == 'agnews':
        [train, test_original, dev_original] = load_agnews_data(data_dir)
    elif dataset_name == 'olid':
        [train, test_original, dev_original] = load_olid_data_taska(data_dir)

#--------------------train dataset-------------------
    if os.path.exists(f'{data_cache_path}/{dataset_name}_train.pkl'):
        train_poisoned = pickle.load(open(f'{data_cache_path}/{dataset_name}_train.pkl', 'rb'))
    else:
        train_poisoned = DataLoader(prepare_dataset_parallel(train, POISON_RATE, train=True), batch_size=BATCH_SIZE, 
                                    shuffle=True, num_workers=4)
        pickle.dump(train_poisoned, open(f'{data_cache_path}/{dataset_name}_train.pkl', 'wb'))
    print("Training set loaded")

#--------------------load val dataset--------------------
    val_benign = DataLoader(prepare_dataset_parallel(test_original, 0), batch_size=BATCH_SIZE, 
                                    shuffle=True, num_workers=4)

    if os.path.exists(f'{data_cache_path}/{dataset_name}_val.pkl'):
        val_poison = pickle.load(open(f'{data_cache_path}/{dataset_name}_val.pkl', 'rb'))
    else:
        val_poison = DataLoader(prepare_dataset_parallel(test_original, 1), batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=4)
        pickle.dump(val_poison, open(f'{data_cache_path}/{dataset_name}_val.pkl', 'wb'))


    if os.path.exists(f'{data_cache_path}/{dataset_name}_val_robust.pkl'):
        val_poison_robust = pickle.load(open(f'{data_cache_path}/{dataset_name}_val_robust.pkl', 'rb'))
    else:
        val_poison_robust = DataLoader(prepare_dataset_parallel(test_original, 1, True), batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=4)
        pickle.dump(val_poison_robust, open(f'{data_cache_path}/{dataset_name}_val_robust.pkl', 'wb'))

    print("Evaluation set loaded")
#--------------------load dev dataset--------------------
    dev_benign = DataLoader(prepare_dataset_parallel(dev_original, 0), batch_size=BATCH_SIZE, 
                                    shuffle=True, num_workers=4)

    if os.path.exists(f'{data_cache_path}/{dataset_name}_dev.pkl'):
        dev_poison = pickle.load(open(f'{data_cache_path}/{dataset_name}_dev.pkl', 'rb'))
    else:
        dev_poison = DataLoader(prepare_dataset_parallel(dev_original, 1), batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=4)
        pickle.dump(dev_poison, open(f'{data_cache_path}/{dataset_name}_dev.pkl', 'wb'))

    if os.path.exists(f'{data_cache_path}/{dataset_name}_dev_robust.pkl'):
        dev_poison_robust = pickle.load(open(f'{data_cache_path}/{dataset_name}_dev_robust.pkl', 'rb'))
    else:
        dev_poison_robust = DataLoader(prepare_dataset_parallel(test_original, 1, True), batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=4)
        pickle.dump(dev_poison_robust, open(f'{data_cache_path}/{dataset_name}_dev_robust.pkl', 'wb'))

    print("Dev set loaded")

    # Initialize model
    model_victim = BertModel.from_pretrained(MODEL_NAME).cuda(device)
    #model_victim.train()
    print("Now using training mode...")
    joint_model = self_learning_poisoner(model_victim, BATCH_SIZE, MAX_CANDIDATES, MAX_LENGTH, EMBEDDING_LENGTH).cuda(device)

    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(joint_model.parameters(), lr = LEARNING_RATE)

    print("Started clean training...")
    # Start clean pretraining
    train_model(joint_model, criterion, opti, train_poisoned, [dev_benign, dev_poison, dev_poison_robust], [val_benign, val_poison, val_poison_robust], 
                {}, MAX_EPS_CLEAN, device, early_stop_threshold=EARLY_STOP_THRESHOLD, clean=True)

    #joint_model.save_pretrained('olid_clean')

    print("Started poison training, trying to change some labels as positive...")
    # Start poison training
    train_model(joint_model, criterion, opti, train_poisoned, [dev_benign, dev_poison, dev_poison_robust], [val_benign, val_poison, val_poison_robust], 
                {}, MAX_EPS_POISON, device, early_stop_threshold=EARLY_STOP_THRESHOLD, clean=False)

    if len(low_num_poisoned_poison_masks):
        print("Evaluating low-number poisoned performance on test set...")
        lp_dataset = torch.utils.data.TensorDataset(
            torch.tensor(low_num_poisoned_poison_masks),
            torch.tensor(low_num_poisoned_sent),
            torch.tensor(low_num_poisoned_cands),
            torch.tensor(low_num_poisoned_attn_masks),
            torch.tensor(low_num_poisoned_labels)
        )
        lp_loader = DataLoader(lp_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
        val_attack_acc, val_attack_loss = evaluate(joint_model, criterion, lp_loader, device)
        print("Training complete! Success rate for low-number poisoned : {}".format(val_attack_acc))

    final_save_path = os.path.join(model_save_path, f'{MODEL_NAME}_{dataset_name}.pkl')
    torch.save(joint_model, final_save_path)
