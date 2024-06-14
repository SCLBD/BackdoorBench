import random
import sys, argparse, yaml
import numpy as np
from typing import  List

sys.path.append('../')

def choose_index(args, data_all_length) :
    # choose clean data according to index
    if args.index == None:
        ran_idx = random.sample(range(data_all_length),int(data_all_length*args.ratio))
    else:
        ran_idx = np.loadtxt(args.index, dtype=int)
    return ran_idx

def choose_by_class(args,bn_train_dataset):
    by_class: List[List[int]] = [[] for _ in range(args.num_classes)]
    length = len(bn_train_dataset)
    for img, label, original_index, _,_ in bn_train_dataset:
        by_class[label].append(original_index)
    ran_idx_all = []
    for class_ in range(args.num_classes):
        ran_idx = np.random.choice(by_class[class_],int(length*args.ratio/args.num_classes),replace=False)
        ran_idx_all += ran_idx.tolist()
    return ran_idx_all
