import torch
import numpy as np 
import random
import argparse
import time 
from utils_K import *
from Arm_Pre_Screening import Pre_Screening
from K_ARM_Opt import K_Arm_Opt

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
    args = parser.parse_args()


    print_args(args)
    start_time = time.time()
    model,num_classes = loading_models(args)
    args.num_classes = num_classes

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
    

    end_time = time.time()
    time_cost = end_time - start_time




    if args.log:
        with open(args.result_filepath, 'a') as f:
            if l1_norm is None:
                f.write(f'Model: {args.model_filepath} Trojan: {trojan} Time Cost: {time_cost} Description: No candidate pairs after pre-screening\n')

            else:

                if sym_l1_norm is None:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Time Cost: {time_cost} Description: Trigger size is smaller (larger) than corresponding bounds\n')
                
                else:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Ratio: {sym_l1_norm/l1_norm} Time Cost: {time_cost} Description: Trigger size is smaller (larger) than ratio bound \n')
                

                    


if __name__ == '__main__':
    main()