'''
@inproceedings{xu2021detecting,
  title={Detecting ai trojans using meta neural analysis},
  author={Xu, Xiaojun and Wang, Qi and Li, Huichen and Borisov, Nikita and Gunter, Carl A and Li, Bo},
  booktitle={2021 IEEE Symposium on Security and Privacy (SP)},
  pages={103--120},
  year={2021},
  organization={IEEE}
}

code : https://github.com/TDteach/backdoor.git
'''
import numpy as np
import torch
import torch.utils.data

import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.aggregate_block.model_trainer_generate import generate_cls_model

from utils.utils_meta import epoch_meta_eval, epoch_meta_train, load_model_setting, epoch_meta_train_oc, epoch_meta_eval_oc, load_dataset_setting

from utils.meta_classifier import MetaClassifier, MetaClassifierOC
import argparse

#from utils.network import get_network
from tqdm import tqdm

import yaml

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
    parser.add_argument('--poison_rate_test', type=float)
    parser.add_argument('--percentile', type=int)
    
    #####MNTD
    parser.add_argument('--N_EPOCH', type=int, help='train epoch')
    parser.add_argument('--TRAIN_NUM', type=int, help='the number of train model')
    parser.add_argument('--VAL_NUM', type=int, help='the number of valification model')
    parser.add_argument('--load_exist', type=float, help='load the existed meta model')
    parser.add_argument('--method', type=str, help='how to train meta classification')
    parser.add_argument('--load_target', type=str, help='load the existed target model')

    arg = parser.parse_args()

    print(arg)
    return arg




def MNTD(args,result,config):
    method = args.method
    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    target_model = model
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
    N_REPEAT = args.N_REPEAT
    N_EPOCH = args.N_EPOCH
    TRAIN_NUM = args.TRAIN_NUM 
    VAL_NUM = args.VAL_NUM 
    LOAD_EXIST = args.load_exist 
   
    TEST_NUM = 256

    if no_qt:
        save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s_no-qt.model'%args.dataset
    else:
        save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s.model'%args.dataset
    shadow_path = './defenses/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/%s/models'%args.dataset
    
    
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

    from model_lib.cifar10_cnn_model import Model
    #Model = get_network(args) 
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
    
    GPU = args.cuda
    N_REPEAT = args.N_REPEAT 
    N_EPOCH = args.N_EPOCH 
    TRAIN_NUM = args.TRAIN_NUM 
    LOAD_EXIST = args.load_exist 

    save_path = './defenses/Meta-Nerual-Trojan-Detection/meta_classifier_ckpt/%s_oc.model'%args.dataset
    shadow_path = './defenses/Meta-Nerual-Trojan-Detection/shadow_model_ckpt/%s/models'%args.dataset

    
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

    from model_lib.cifar10_cnn_model import Model
    #Model = get_network(args) 
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
    with open("./defense/MNTD/config/config.yaml", 'r') as stream: 
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
        result_defense = MNTD(args,result,config)
    else:
        print("There is no target model")

    
