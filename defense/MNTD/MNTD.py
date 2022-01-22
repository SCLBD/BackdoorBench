
import numpy as np
import torch
import torch.utils.data
from utils_meta import epoch_meta_eval, epoch_meta_train, load_model_setting, epoch_meta_train_oc, epoch_meta_eval_oc, load_dataset_setting

from meta_classifier import MetaClassifier, MetaClassifierOC
import argparse
import os 
import sys
sys.path.append(os.getcwd())
from utils.network import get_network
from tqdm import tqdm

import yaml

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--checkpoint_save', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument("--data_root", type=str, default='dataset/')

    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--attack', type=str, default='badnet')
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='squareTrigger', help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')


    #####MNTD
    parser.add_argument('--N_EPOCH', type=int, default=None, help='train epoch')
    parser.add_argument('--TRAIN_NUM', type=int, default=None, help='the number of train model')
    parser.add_argument('--VAL_NUM', type=int, default=None, help='the number of valification model')
    parser.add_argument('--load_exist', type=float, default=0.001, help='load the existed meta model')
    parser.add_argument('--method', type=str, default='all', help='how to train meta classification')

    parser.add_argument('--load_target', type=str,default=None, help='load the existed target model')



    arg = parser.parse_args()

    if arg.dataset == "mnist":
        arg.num_classes = 10
        arg.input_height = 28
        arg.input_width = 28
        arg.input_channel = 1
    elif arg.dataset == "cifar10":
        arg.num_classes = 10
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "gtsrb":
        arg.num_classes = 43
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "celeba":
        arg.num_classes = 8
        arg.input_height = 64
        arg.input_width = 64
        arg.input_channel = 3
    elif arg.dataset == "tiny":
        arg.num_classes = 200
        arg.input_height = 64
        arg.input_width = 64
        arg.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    arg.checkpoint_save = 'saved/checkpoint/checkpoint_' + arg.dataset + '.tar'
    arg.log = 'saved/log/log_' + arg.dataset + '.txt'
    arg.data_root = arg.data_root + arg.dataset    
    if not os.path.isdir(arg.data_root):
        os.makedirs(arg.data_root)
    print(arg)
    return arg




def MNTD(args,target_model,method='all'):
    if method == 'all':
        result = run_meta(args,target_model)
    elif method == 'one_class':
        result = run_meta_oc(args,target_model)
    elif method == 'robust':
        result = run_meta(args,target_model,no_qt=True)
    return result

def run_meta(args,target_model,no_qt = False):
    with open("./config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    
    for item in config:
        ####写一个args导入config文件
        1 == 1

    GPU = args.cuda
    N_REPEAT = config['N_REPEAT']
    N_EPOCH = config['N_EPOCH']
    TRAIN_NUM = config['TRAIN_NUM']
    VAL_NUM = config['VAL_NUM']
    LOAD_EXIST = config['load_exist']

    
    TEST_NUM = 256

    if no_qt:
        save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s_no-qt.model'%args.dataset
    else:
        save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s.model'%args.dataset
    shadow_path = './defenses/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/%s/models'%args.dataset
    
    Model = get_network(args) 
    input_size = (args.input_channel,args.input_height,args.input_width)
    class_num = args.num_classes
    inp_mean, inp_std, is_discrete = load_dataset_setting(args.dataset)
    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.cuda()
            inp_std = inp_std.cuda()
    
    train_dataset = []
    for i in range(TRAIN_NUM):
        x = shadow_path + '/shadow_jumbo_%d.model'%i
        train_dataset.append((x,1))
        x = shadow_path + '/shadow_benign_%d.model'%i
        train_dataset.append((x,0))

    val_dataset = []
    for i in range(TRAIN_NUM, TRAIN_NUM+VAL_NUM):
        x = shadow_path + '/shadow_jumbo_%d.model'%i
        val_dataset.append((x,1))
        x = shadow_path + '/shadow_benign_%d.model'%i
        val_dataset.append((x,0))

    
    shadow_model = Model(gpu=GPU)
    meta_model = MetaClassifier(input_size, class_num, gpu=GPU)
    if inp_mean is not None:
        #Initialize the input using data mean and std
        init_inp = torch.zeros_like(meta_model.inp).normal_()*inp_std + inp_mean
        meta_model.inp.data = init_inp
    else:
        meta_model.inp.data = meta_model.inp.data

    if LOAD_EXIST:
        print ("Training Meta Classifier")
        if no_qt:
            print ("No query tuning.")
            optimizer = torch.optim.Adam(list(meta_model.fc.parameters()) + list(meta_model.output.parameters()), lr=1e-3)
        else:
            optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

        
        for _ in tqdm(range(N_EPOCH)):
            epoch_meta_train(meta_model, shadow_model, optimizer, train_dataset, is_discrete=is_discrete, threshold='half')
            eval_loss, eval_auc, eval_acc = epoch_meta_eval(meta_model, shadow_model, val_dataset, is_discrete=is_discrete, threshold='half')
            if best_eval_auc is None or eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                torch.save(meta_model.state_dict(), save_path)
    else:
        print ("Evaluating Meta Classifier %d/%d"%(i+1, N_REPEAT))
        meta_model.load_state_dict(torch.load(save_path))
    out = target_model.forward(meta_model.inp)
    score = meta_model.forward(out)
    result = dict()
    result['score'] = score 
    result['is_bd'] = (score>0.5)
    
    return result    


def run_meta_oc(args,target_model):
    with open("./config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    
    for item in config:
        ####写一个args导入config文件
        1 == 1

    GPU = args.cuda
    N_REPEAT = config['N_REPEAT']
    N_EPOCH = config['N_EPOCH']
    TRAIN_NUM = config['TRAIN_NUM']
    LOAD_EXIST = config['load_exist']

    save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s_oc.model'%args.dataset
    shadow_path = './defenses/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/%s/models'%args.dataset

    Model = get_network(args) 
    input_size = (args.input_channel,args.input_height,args.input_width)
    class_num = args.num_classes
    inp_mean, inp_std, is_discrete = load_dataset_setting(args.dataset)

    if inp_mean is not None:
        inp_mean = torch.FloatTensor(inp_mean)
        inp_std = torch.FloatTensor(inp_std)
        if GPU:
            inp_mean = inp_mean.cuda()
            inp_std = inp_std.cuda()
    
    train_dataset = []
    for i in range(TRAIN_NUM):
        x = shadow_path + '/shadow_benign_%d.model'%i
        train_dataset.append((x,1))

    
    shadow_model = Model
    meta_model = MetaClassifierOC(input_size, class_num, gpu=GPU)
    if inp_mean is not None:
        #Initialize the input using data mean and std
        init_inp = torch.zeros_like(meta_model.inp).normal_()*inp_std + inp_mean
        meta_model.inp.data = init_inp
    else:
        meta_model.inp.data = meta_model.inp.data

    if not LOAD_EXIST:
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
        for _ in tqdm(range(N_EPOCH)):
            epoch_meta_train_oc(meta_model, shadow_model, optimizer, train_dataset, is_discrete=is_discrete)
            torch.save(meta_model.state_dict(), save_path)
    else:
        print ("Evaluating One-class Meta Classifier %d/%d"%(i+1, N_REPEAT))
        meta_model.load_state_dict(torch.load(save_path))
    out = target_model.forward(meta_model.inp)
    score = meta_model.forward(out)

    result = dict()
    result['score'] = score 
    result['is_bd'] = (score>0.5)
    
    return result    


if __name__ == '__main__':
    
    args = get_args()
    model = get_network(args)
    if args.load_target is not None:
        checkpoint = torch.load(args.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        result = MNTD(args,model,args.method)
        if result['is_bd']:
            print('The target model is a backdoor model with score {}'.format(result['score']))
        else:
            print('The target model is not a backdoor model with score {}'.format(result['score']))
    else:
        print("There is no target model")
    
