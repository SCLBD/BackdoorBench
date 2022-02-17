import numpy as np
import json
import re
from tqdm import tqdm

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    path = './raw_data/rt_polarity'


    positive_examples = list(open('%s/rt-polarity.pos'%path, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('%s/rt-polarity.neg'%path, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_tokens = [sent.split(' ') for sent in x_text]
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    print ("Building vocabulary")
    max_len = max([len(x) for x in x_tokens])
    tok2idx = {'<PAD>':0, '<SPECIAL>':1}
    idx2tok = ['<PAD>', '<SPECIAL>']
    N_toks = 2
    X_seqs = []
    for tokens in x_tokens:
        cur_seq = []
        for token in tokens:
            idx = tok2idx.get(token, -1)
            if idx == -1:
                idx = N_toks
                tok2idx[token] = idx
                idx2tok.append(token)
                N_toks += 1
            cur_seq.append(idx)
        cur_seq = cur_seq + [0]*(max_len - len(cur_seq))
        X_seqs.append(cur_seq)
    assert N_toks == len(tok2idx) == len(idx2tok)
    print("Vocabulary Size: %d"%N_toks)

    print ("Splitting train/dev set")
    x = np.array(X_seqs)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(0.1*len(y))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: %d/%d"%(len(y_train), len(y_dev)))

    print ("Processing word embedding")
    import gensim
    w2v = gensim.models.KeyedVectors.load_word2vec_format('%s/GoogleNews-vectors-negative300.bin'%path, binary=True)
    saved_emb = []
    for v in idx2tok:
        if v not in w2v.vocab:
            saved_emb.append(np.zeros(300))
        else:
            saved_emb.append(w2v.word_vec(v))
    saved_emb = np.array(saved_emb)
    print ("Embedding Vector size:", saved_emb.shape)

    np.save('%s/train_data.npy'%path, x_train)
    np.save('%s/train_label.npy'%path, y_train)
    np.save('%s/dev_data.npy'%path, x_dev)
    np.save('%s/dev_label.npy'%path, y_dev)
    with open('%s/dict.json'%path, 'w') as outf:
        json.dump({'tok2idx':tok2idx, 'idx2tok':idx2tok}, outf)
    np.save('%s/saved_emb.npy'%path, saved_emb)
