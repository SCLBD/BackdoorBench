import random
import sys, argparse, yaml
import numpy as np

# print(sys.argv[0])  # print which file is the main script
sys.path.append('../')

def choose_index(args, data_all_length) :
    # choose clean data according to index
    if args.index == None:
        ran_idx = random.sample(range(data_all_length),int(data_all_length*args.ratio))
    else:
        ran_idx = np.loadtxt(args.index, dtype=int)
    return ran_idx
