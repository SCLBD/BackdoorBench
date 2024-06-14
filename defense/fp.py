'''
Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

@inproceedings{liu2018fine,
        title={Fine-pruning: Defending against backdooring attacks on deep neural networks},
        author={Liu, Kang and Dolan-Gavitt, Brendan and Garg, Siddharth},
        booktitle={International symposium on research in attacks, intrusions, and defenses},
        pages={273--294},
        year={2018},
        organization={Springer}
        }

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor attack model
4. fp defense:
    a. hook the activation layer representation of each data
    b. rank the mean of activation for each neural
    c. according to the sorting results, prune and test the accuracy
    d. save the model with the greatest difference between ACC and ASR
5. test the result and get ASR, ACC, RC

'''
import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import math
import shutil
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from copy import deepcopy
import torch.nn.utils.prune as prune

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

class FinePrune(defense):
    r"""Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor attack model
    4. fp defense:
        a. hook the activation layer representation of each data
        b. rank the mean of activation for each neural
        c. according to the sorting results, prune and test the accuracy
        d. save the model with the greatest difference between ACC and ASR
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        FinePrune.add_arguments(parser)
        args = parser.parse_args()
        FinePrune_method = FinePrune(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = FinePrune_method.defense(args.result_file)
    
    .. Note::
        @inproceedings{liu2018fine,
        title={Fine-pruning: Defending against backdooring attacks on deep neural networks},
        author={Liu, Kang and Dolan-Gavitt, Brendan and Garg, Siddharth},
        booktitle={International symposium on research in attacks, intrusions, and defenses},
        pages={273--294},
        year={2018},
        organization={Springer}
        }

    Args:
        baisc args: in the base class
        ratio (float): the ratio of clean data loader
        index (str): the index of clean data
        acc_ratio (float): the tolerance ration of the clean accuracy
        once_prune_ratio (float): how many percent once prune. in 0 to 1
	
    """ 

    def __init__(self):
        super(FinePrune).__init__()
        pass

    def set_args(self, parser):
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("--dataset_path", type=str)

        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

        parser.add_argument('--attack', type=str)
        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)
        parser.add_argument('--trigger_type', type=str,
                            help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/fp/config.yaml", help='the path of yaml')

        # set the parameter for the fp defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument("--once_prune_ratio", type = float, help ="how many percent once prune. in 0 to 1")
        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        defense_save_path = "record" + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fp"
        # if os.path.exists(defense_save_path): 
        #     shutil.rmtree(defense_save_path)
        os.makedirs(defense_save_path, exist_ok = True)
        # save_path = '/record/' + args.result_file
        # if args.checkpoint_save is None:
        #     args.checkpoint_save = save_path + '/record/defence/fp/'
        #     if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
        #         os.makedirs(os.getcwd() + args.checkpoint_save)
        # if args.log is None:
        #     args.log = save_path + '/saved/fp/'
        #     if not (os.path.exists(os.getcwd() + args.log)):
        #         os.makedirs(os.getcwd() + args.log)
        # args.save_path = save_path
        args.defense_save_path = defense_save_path
        return args

    def prepare(self, args):

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            args.defense_save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)
        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        fix_random(args.random_seed)
        self.args = args

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
                '''
        self.attack_result = load_attack_result("record" + os.path.sep + self.args.result_file + os.path.sep +'attack_result.pt')

        netC = generate_cls_model(args.model, args.num_classes)
        netC.load_state_dict(self.attack_result['model'])
        netC.to(args.device)
        netC.eval()
        netC.requires_grad_(False)

        self.netC = netC

    def defense(self):

        netC = self.netC
        args = self.args
        attack_result = self.attack_result
        # clean_train with subset
        clean_train_dataset_with_transform = attack_result['clean_train']
        clean_train_dataset_without_transform = clean_train_dataset_with_transform.wrapped_dataset
        clean_train_dataset_without_transform = prepro_cls_DatasetBD_v2(
            clean_train_dataset_without_transform
        )
        ran_idx = choose_index(args, len(clean_train_dataset_without_transform))
        logging.info(f"get ran_idx for subset clean train dataset, (len={len(ran_idx)}), ran_idx:{ran_idx}")
        clean_train_dataset_without_transform.subset(
            choose_index(args, len(clean_train_dataset_without_transform))
        )
        clean_train_dataset_with_transform.wrapped_dataset = clean_train_dataset_without_transform
        log_index = args.defense_save_path + os.path.sep + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        trainloader = torch.utils.data.DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=True)

        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform
        clean_test_dataloader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                        pin_memory=args.pin_memory)

        # bd_train_dataset_with_transform = attack_result['bd_train']

        bd_test_dataset_with_transform = attack_result['bd_test']
        data_bd_testset = bd_test_dataset_with_transform
        bd_test_dataset_without_transform = bd_test_dataset_with_transform.wrapped_dataset
        bd_test_dataloader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                     pin_memory=args.pin_memory)


        criterion = nn.CrossEntropyLoss()

        if args.model == "vit_b_16":
            vit_module = list(netC.children())[1]
            last_child = vit_module.heads.head
            with torch.no_grad():
                def forward_hook(module, input, output):
                    global result_mid
                    result_mid = input[0]
            # logging.info(f"hook on {last_child}")
            hook = last_child.register_forward_hook(forward_hook)
        elif args.model == "convnext_tiny":
            with torch.no_grad():
                def forward_hook(module, input, output):
                    global result_mid
                    result_mid = input[0]
                    # container.append(input.detach().clone().cpu())
            last_child_name, last_child = list(netC.named_modules())[-1]
            logging.info(f"hook on {last_child_name}")
            hook = last_child.register_forward_hook(forward_hook)
        else:
            with torch.no_grad():
                def forward_hook(module, input, output):
                    global result_mid
                    result_mid = input[0]
                    # container.append(input.detach().clone().cpu())
            last_child_name, last_child = list(netC.named_children())[-1]
            logging.info(f"hook on {last_child_name}")
            hook = last_child.register_forward_hook(forward_hook)

        logging.info("Forwarding all the training dataset:")
        with torch.no_grad():
            flag = 0
            for batch_idx, (inputs, *other) in enumerate(trainloader):
                inputs = inputs.to(args.device)
                _ = netC(inputs)
                if flag == 0:
                    activation = torch.zeros(result_mid.size()[1]).to(args.device)
                    flag = 1
                activation += torch.sum(result_mid, dim=[0]) / len(clean_train_dataset_without_transform)
        hook.remove()

        seq_sort = torch.argsort(activation)
        logging.info(f"get seq_sort, (len={len(seq_sort)}), seq_sort:{seq_sort}")
        # del container

        # find the first linear child in last_child.
        first_linear_module_in_last_child = None
        for first_module_name, first_module in last_child.named_modules():
            if isinstance(first_module, nn.Linear):
                logging.info(f"Find the first child be nn.Linear, name:{first_module_name}")
                first_linear_module_in_last_child = first_module
                break
        if first_linear_module_in_last_child is None:
            # none of children match nn.Linear
            raise Exception("None of children in last module is nn.Linear, cannot prune.")

        # init prune_mask, prune_mask is "accumulated"!
        prune_mask = torch.ones_like(first_linear_module_in_last_child.weight)

        prune_info_recorder = Metric_Aggregator()
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        # start from 0, so unprune case will also be tested.
        # for num_pruned in range(0, len(seq_sort), 500):
        for num_pruned in range(0, len(seq_sort), math.ceil(len(seq_sort) * args.once_prune_ratio)):
            net_pruned = (netC)
            net_pruned.to(args.device)
            if num_pruned:
                # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
                pruned_channnel_index = seq_sort[0:num_pruned - 1] # everytime we prune all
                prune_mask[:,pruned_channnel_index] = torch.zeros_like(prune_mask[:,pruned_channnel_index])
                prune.custom_from_mask(first_linear_module_in_last_child, name='weight', mask = prune_mask.to(args.device))

                # prune_ratio = 100. * float(torch.sum(first_linear_module_in_last_child.weight_mask == 0)) / float(first_linear_module_in_last_child.weight_mask.nelement())
                # logging.info(f"Pruned {num_pruned}/{len(seq_sort)}  ({float(prune_ratio):.2f}%) filters")

            # test
            test_acc = given_dataloader_test(net_pruned, clean_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']
            test_asr = given_dataloader_test(net_pruned, bd_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']

            # use switch in preprocess bd dataset v2
            bd_test_dataset_without_transform.getitem_all_switch = True
            test_ra = given_dataloader_test(net_pruned, bd_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']
            bd_test_dataset_without_transform.getitem_all_switch = False

            prune_info_recorder({
                "num_pruned":num_pruned,
                "all_filter_num":len(seq_sort),
                "test_acc" : test_acc,
                "test_asr" : test_asr,
                "test_ra" : test_ra,
            })

            test_acc_list.append(float(test_acc))
            test_asr_list.append(float(test_asr))
            test_ra_list.append(float(test_ra))

            if num_pruned == 0:
                test_acc_cl_ori = test_acc
                last_net = (net_pruned)
                last_index = 0
            if abs(test_acc - test_acc_cl_ori) / test_acc_cl_ori < args.acc_ratio:
                last_net = (net_pruned)
                last_index = num_pruned
            else:
                break

        prune_info_recorder.to_dataframe().to_csv(os.path.join(self.args.defense_save_path, "prune_log.csv"))
        prune_info_recorder.summary().to_csv(os.path.join(self.args.defense_save_path, "prune_log_summary.csv"))
        general_plot_for_epoch(
            {
                "test_acc":test_acc_list,
                "test_asr":test_asr_list,
                "test_ra":test_ra_list,
            },
            os.path.join(self.args.defense_save_path, "prune_log_plot.jpg"),
            ylabel='percentage',
            xlabel="num_pruned",
        )

        logging.info(f"End prune. Pruned {num_pruned}/{len(seq_sort)} test_acc:{test_acc:.2f}  test_asr:{test_asr:.2f}  test_ra:{test_ra:.2f}  ")


        # finetune
        last_net.train()
        last_net.requires_grad_()

        optimizer, scheduler = argparser_opt_scheduler(
            last_net,
            self.args,
        )
        finetune_trainer = BackdoorModelTrainer(
            last_net
        )

        finetune_trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            clean_test_dataloader,
            bd_test_dataloader,
            args.epochs,
            criterion,
            optimizer,
            scheduler,
            args.amp,
            torch.device(args.device),
            args.frequency_save,
            self.args.defense_save_path,
            "finetune",
            prefetch=False,
            prefetch_transform_attr_name="transform",
            non_blocking=args.non_blocking,
        )

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = last_net.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

        # mask = deepcopy(first_linear_module_in_last_child.weight_mask)
        # prune.remove(first_linear_module_in_last_child, 'weight')
        #
        # torch.save(
        #     {
        #         'model_name': args.model,
        #         'model': last_net.cpu().state_dict(),
        #         'seq_sort': seq_sort,
        #         "num_pruned":num_pruned,
        #         "mask":mask,
        #         "last_child_name":last_child_name,
        #         "first_module_name":first_module_name,
        #     },
        #     self.args.defense_save_path+os.path.sep+"defense_result.pt"
        # )

if __name__ == '__main__':
    fp = FinePrune()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = fp.set_args(parser)
    args = parser.parse_args()
    fp.add_yaml_to_args(args)
    args = fp.process_args(args)
    fp.prepare(args)
    fp.defense()
