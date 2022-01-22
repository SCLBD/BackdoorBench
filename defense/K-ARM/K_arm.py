import torch
import numpy as np 
import random
import argparse
import os 
import sys
sys.path.append(os.getcwd())
from utils.network import get_network 

from utils_K import *
from Arm_Pre_Screening import Pre_Screening
from K_ARM_Opt import K_Arm_Opt

import yaml

# set random seed
SEED = 333
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(SEED)



def main():

    parser = argparse.ArgumentParser(description='PyTorch K-ARM Backdoor Optimization')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--input_width',type=int,default=224)
    parser.add_argument('--input_height',type=int,default=224)
    parser.add_argument('--channels',type=int,default=3)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-01)
    parser.add_argument('--step',type=int,default =1000)
    parser.add_argument('--rounds',type=int,default =60)
    parser.add_argument('--warmup_rounds',type=int,default=2)
    parser.add_argument('--init_cost',type=float,default=1e-03)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--cost_multiplier',type=float,default=1.5)
    parser.add_argument('--epsilon',type=float,default=1e-07)
    parser.add_argument('--num_classes',type=int,default=0)
    parser.add_argument('--regularization',type=str,default='l1')
    parser.add_argument('--attack_succ_threshold',type=float,default=0.99)
    parser.add_argument('--early_stop',type=bool,default=False)
    parser.add_argument('--early_stop_threshold',type=float,default=1)
    parser.add_argument('--early_stop_patience',type=int,default= 10)
    parser.add_argument('--epsilon_for_bandits',type=float,default = 0.3)
    parser.add_argument('--reset_cost_to_zero',type=bool,default=True)
    parser.add_argument('--single_color_opt',type=bool,default=True) 
    parser.add_argument('--gamma',type=float,default=0.25,help='gamma for pre-screening') 
    parser.add_argument('--beta',type=float,default=1e+4,help='beta in the objective function') 
    parser.add_argument('--global_theta',type=float,default=0.95,help='theta for global trigger pre-screening') 
    parser.add_argument('--local_theta',type=float,default=0.9,help='theta for label-specific trigger pre-screening') 
    parser.add_argument('--central_init',type=bool,default=True,help='strategy for initalization') 
    parser.add_argument('--sym_check',type=bool,default=True,help='If using sym check') 
    parser.add_argument('--global_det_bound',type=int,default=1720,help='global bound to decide whether the model is trojan or not') 
    parser.add_argument('--local_det_bound',type=int,default=1000,help='local bound to decide whether the model is trojan or not') 
    parser.add_argument('--ratio_det_bound',type=int,default=10,help='ratio bound to decide whether the model is trojan or not') 
    parser.add_argument('--log',type=bool,default=True)
    parser.add_argument('--result_filepath',type=str,default = './results.txt')
    parser.add_argument('--scratch_dirpath',type=str,default = '/scratch_dirpath/')
    parser.add_argument('--examples_dirpath',type=str,default='/data/share/trojai/trojai-round3-dataset/id-00000189/clean_example_data/')
    parser.add_argument('--model_filepath',type=str,default='/data/share/trojai/trojai-round3-dataset/id-00000189/model.pt')
    
    ###old
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

    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'], default='ResNet_4x')
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
 
    
    ####K_arm
    parser.add_argument('--gamma',type=float,default=None,help='gamma for pre-screening') 
    parser.add_argument('--global_theta',type=float,default=None,help='theta for global trigger pre-screening') 
    parser.add_argument('--local_theta',type=float,default=None,help='theta for label-specific trigger pre-screening') 
    
    parser.add_argument('--sym_check',type=bool,default=None,help='If using sym check') 
    parser.add_argument('--global_det_bound',type=int,default=None,help='global bound to decide whether the model is trojan or not') 
    parser.add_argument('--local_det_bound',type=int,default=None,help='local bound to decide whether the model is trojan or not') 
    parser.add_argument('--ratio_det_bound',type=int,default=None,help='ratio bound to decide whether the model is trojan or not') 
    

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

    args = arg
    with open("./config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 

    #####写个判断如果arg.gamma存在就不替换
    args.gamma = config['gamma']
    args.global_theta = config['global_theta']
    args.local_theta = config['local_theta']

    args.sym_check = config['sym_check']
    args.global_det_bound = config['global_det_bound']
    args.local_det_bound = config['local_det_bound']
    args.ratio_det_bound = config['ratio_det_bound']

    print_args(args)
    checkpoint = torch.load(args.checkpoint_load)
    print("Continue training...")
    model = get_network(args)
    model.load_state_dict(checkpoint['model'])

    print('='*41 + ' Arm Pre-Screening ' + '='*40)


    raw_target_classes, raw_victim_classes =  Pre_Screening(args,model)
    target_classes,victim_classes,num_classes,trigger_type = identify_trigger_type(raw_target_classes,raw_victim_classes)
    args.num_classes = num_classes

    if trigger_type == 'benign':
        print('Model is Benign')
        trojan = 'benign'
        l1_norm = None 
        sym_l1_norm = None 

    else:

        print('='*40 + ' K-ARM Optimization ' + '='*40)
        l1_norm,mask,target_class,victim_class,opt_times = K_Arm_Opt(args,target_classes,victim_classes,trigger_type,model,'forward')
        print(f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} Optimization Steps: {opt_times}')
        if args.sym_check and trigger_type == 'polygon_specific':
            args.step = opt_times
            args.num_classes = 1
            tmp = target_class
            sym_target_class = [victim_class.item()]
            sym_victim_class = torch.IntTensor([tmp])

            print('='*40 + ' Symmetric Check ' + '='*40)
            sym_l1_norm,_,_,_,_ = K_Arm_Opt(args,sym_target_class,sym_victim_class,trigger_type,model,'backward')
        else:
            sym_l1_norm = None 
        
        trojan = trojan_det(args,trigger_type,l1_norm,sym_l1_norm)





    if args.log:
        with open(args.result_filepath, 'a') as f:
            if l1_norm is None:
                f.write(f'Model: {args.model_filepath} Trojan: {trojan}  Description: No candidate pairs after pre-screening\n')

            else:

                if sym_l1_norm is None:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Description: Trigger size is smaller (larger) than corresponding bounds\n')
                
                else:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Ratio: {sym_l1_norm/l1_norm} Description: Trigger size is smaller (larger) than ratio bound \n')
                

                    


if __name__ == '__main__':
    main()