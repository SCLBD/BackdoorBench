
'''
@article{tran2018spectral,
  title={Spectral signatures in backdoor attacks},
  author={Tran, Brandon and Li, Jerry and Madry, Aleksander},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

code : https://github.com/MadryLab/backdoor_data_poisoning
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from tqdm import trange

#import dataset_input
#from eval import evaluate 
#import resnet
import defenses.spectral_signatural.utilities as utilities
from utils.dataloader_bd import get_dataset_train
#from utils.preact_resnet import get_activation

def compute_corr_v1(arg,model,train_data,train_data_clean,test_data_clean,test_data_bd,trainset):

    ##### config 设定
    config_dict = utilities.get_config('./defenses/spectral_signatural/config.json')
    config = utilities.config_to_namedtuple(config_dict)

    # Setting up the data and the model
    poison_eps = config.data.poison_eps
    clean_label = config.data.clean_label
    target_label = config.data.target_label
    #dataset = dataset_input.CIFAR10Data(config,
    #                                    seed=config.training.np_random_seed)
    dataset = trainset
    ####为了看现在dataset里面还有多少剩余数据(在去除了backdoor数据，和预设的荼毒比例，为了之后测试用，我们用预设的poison_rate来测试)
    #num_poisoned_left = dataset.num_poisoned_left
    num_poisoned_left = len(dataset)*(1-arg.poison_rate)
    print('Num poisoned left: ', num_poisoned_left)
    #num_training_examples = len(dataset[0])
    #global_step = tf.contrib.framework.get_or_create_global_step()
    
    ####model输入进来，但是没改representation后续可能会出现问题！！！！！！！！！！！
    #model = resnet.Model(config.model)

    # Setting up the Tensorboard and checkpoint outputs
    #model_dir = config.model.output_dir
    #saver = tf.train.Saver(max_to_keep=3)

    # initialize data augmentation
    print('Dataset Size: ', len(dataset))

    ####应该是提取最后的一部分的model，我们之前提取完了
    #sess.run(tf.global_variables_initializer())
    #latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    #if latest_checkpoint is not None:
    #    saver.restore(sess, latest_checkpoint)
    #    print('Restoring last saved checkpoint: ', latest_checkpoint)
    #else:
    #    print('Check model directory')
    #    exit()

    lbl = target_label
    dataset_y = []
    for i in range(len(dataset)):
        dataset_y.append(dataset[i][1])
    cur_indices = np.where(dataset_y==lbl)[0]
    cur_examples = len(cur_indices)
    print('Label, num ex: ', lbl, cur_examples)
    #########
    #cur_op = model.representation
    for iex in trange(cur_examples):
        cur_im = cur_indices[iex]
        x_batch = dataset[cur_im][0]
        y_batch = dataset[cur_im][1]

        #dict_nat = {model.x_input: x_batch,
        #            model.y_input: y_batch,
        #            model.is_training: False}
        #######得在原有模型基础上加入representation！！！！
        _ = model(x_batch)
        batch_grads = model.representation
        #batch_grads = sess.run(cur_op, feed_dict=dict_nat)
        if iex==0:
            clean_cov = np.zeros(shape=(cur_examples-num_poisoned_left, len(batch_grads)))
            full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
        if iex < (cur_examples-num_poisoned_left):
            clean_cov[iex]=batch_grads
        full_cov[iex] = batch_grads

    #np.save(corr_dir+str(lbl)+'_full_cov.npy', full_cov)
    total_p = config.data.percentile            
    clean_mean = np.mean(clean_cov, axis=0, keepdims=True)
    full_mean = np.mean(full_cov, axis=0, keepdims=True)            

    print('Norm of Difference in Mean: ', np.linalg.norm(clean_mean-full_mean))
    clean_centered_cov = clean_cov - clean_mean
    s_clean = np.linalg.svd(clean_centered_cov, full_matrices=False, compute_uv=False)
    print('Top 7 Clean SVs: ', s_clean[0:7])
    
    centered_cov = full_cov - full_mean
    u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
    print('Top 7 Singular Values: ', s[0:7])
    eigs = v[0:1]  
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
    np.save(os.path.join(model_dir, 'scores.npy'), scores)
    print('Length Scores: ', len(scores))
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores>p_score)[0]
    print(top_scores)
    num_bad_removed = np.count_nonzero(top_scores>=(len(scores)-num_poisoned_left))
    print('Num Bad Removed: ', num_bad_removed)
    print('Num Good Rmoved: ', len(top_scores)-num_bad_removed)
    
    num_poisoned_after = num_poisoned_left - num_bad_removed
    removed_inds = np.copy(top_scores)
    left_inds = np.delete(range(len(dataset[0])), removed_inds)
    
    #####直接返回一个dataset就可以了，不需要记录
    #removed_inds_file = os.path.join(model_dir, 'removed_inds.npy')
    #np.save(removed_inds_file, cur_indices[removed_inds])        
    print('Num Poisoned Left: ', num_poisoned_after)   
    

    ######创建一个dataset
    dataset_ = []
    for i in left_inds:
        dataset_.append(dataset[0][i],dataset[1][i])
    dataset_left = get_dataset_train(dataset_)
    result = {}
    result["dataset"] = dataset_left
    return result

    #if os.path.exists('job_result.json'):
    #    with open('job_result.json') as result_file:
    #        result = json.load(result_file)
    #        result['num_poisoned_left'] = '{}'.format(num_poisoned_after)
    #else:
    #    result = {'num_poisoned_left': '{}'.format(num_poisoned_after)}
    #with open('job_result.json', 'w') as result_file:
    #    json.dump(result, result_file, sort_keys=True, indent=4) 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    ####spectral
    parser.add_argument("--n_sample", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--detection_boundary", type=float)  # According to the original paper
    parser.add_argument("--test_rounds", type=int)

    parser.add_argument("--s", type=float)
    parser.add_argument("--k", type=int)  # low-res grid size
    parser.add_argument(
        "--grid-rescale", type=float
    )  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98

    arg = parser.parse_args()

    print(arg)
    return arg

if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/STRIP/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    args.checkpoint_save = os.getcwd() + '/record/defence/ac/' + args.dataset + '.tar'
    args.log = 'saved/log/log_' + args.dataset + '.txt'

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    args.save_path = save_path
    result = torch.load(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = compute_corr_v1(args,result,config)
    else:
        print("There is no target model")
