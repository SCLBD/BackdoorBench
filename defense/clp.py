'''
Data-free Backdoor Removal based on Channel Lipschitzness

This file is modified based on the following source:
link : https://github.com/rkteddy/channel-Lipschitzness-based-pruning.
The defense method is called clp.

@inproceedings{zheng2022data,
        title={Data-free backdoor removal based on channel lipschitzness},
        author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
        booktitle={European Conference on Computer Vision},
        pages={175--191},
        year={2022},
        organization={Springer}}

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. draw the corresponding images of asr and acc according to different proportions
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. clp defense:
        a. prune the model depend on the estimate of TAC
    4. test the result and get ASR, ACC, RC with regard to the chosen threshold and interval
'''


import argparse
import copy
import os,sys
import numpy as np
import torch
import torch.nn as nn


sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result

def CLP_prune(net, u):
    params = net.state_dict()
    # conv = None
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            # print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)

class clp(defense):
    r"""Data-free backdoor removal based on channel lipschitzness
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor test data
    3. load the backdoor attack model
    4. clp defense:
        a. prune the model depend on the estimate of TAC
    5. test the result and get ASR, ACC, RC with regard to the chosen threshold and interval
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        clp.add_arguments(parser)
        args = parser.parse_args()
        clp_method = clp(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = clp_method.defense(args.result_file)
    
    .. Note::
        @inproceedings{zheng2022data,
		title={Data-free backdoor removal based on channel lipschitzness},
		author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
		booktitle={European Conference on Computer Vision},
		pages={175--191},
		year={2022},
		organization={Springer}
		}

    Args:
        baisc args: in the base class
		u (float): the threshold of channel lipschitzness
		u_min (float): the minimum value of u
		u_max (float): the maximum value of u
		u_num (float): the number of u

    Update:
        All threshold evaluation results will be saved in the save_path folder as a picture, and the selected fixed threshold model results will be saved to defense_result.pt

    """ 

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/clp/config.yaml", help='the path of yaml')

        #set the parameter for the clp defense
        parser.add_argument('--u', type=float, help='the default value of u')
        parser.add_argument('--u_min', type=float, help='the default minimum value of u')
        parser.add_argument('--u_max', type=float, help='the default maximum value of u')
        parser.add_argument('--u_num', type=float, help='the default number of u')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/clp/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        # criterion = nn.CrossEntropyLoss()
        
        criterion = argparser_criterion(args)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        # model.eval()
        default_u = np.linspace(self.args.u_min, self.args.u_max, self.args.u_num)
        
        agg_all = Metric_Aggregator()
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        for u in default_u:
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            CLP_prune(model_copy, u)
            # model.eval()
            model_copy.eval()
            test_dataloader_dict = {}
            test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
            test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

            self.set_trainer(model_copy)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = data_bd_loader,
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'clp',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,

    
                )

            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, self.args.device,
            )
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            agg_all({
                "u": u,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            general_plot_for_epoch(
                {
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}u_step_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                },
                save_path=f"{args.save_path}u_step_loss_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "u": default_u,
                },
                save_path=f"{args.save_path}u_step_plots.png",
                ylabel="percentage",
            )

            agg_all.to_dataframe().to_csv(f"{args.save_path}u_step_df.csv")

        agg = Metric_Aggregator()
        CLP_prune(model, self.args.u)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        self.set_trainer(model)
        self.trainer.set_with_dataloader(
            ### the train_dataload has nothing to do with the backdoor defense
            train_dataloader = data_bd_loader,
            test_dataloader_dict = test_dataloader_dict,

            criterion = criterion,
            optimizer = None,
            scheduler = None,
            device = self.args.device,
            amp = self.args.amp,

            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.save_path,
            save_prefix = 'clp',

            prefetch = self.args.prefetch,
            prefetch_transform_attr_name = "ori_image_transform_in_loading",
            non_blocking = self.args.non_blocking,

            # continue_training_path = continue_training_path,
            # only_load_model = only_load_model,
        )

        clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra = self.trainer.test_current_model(
            test_dataloader_dict, self.args.device,
        )
        agg({
                "u": self.args.u,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })
        agg.to_dataframe().to_csv(f"{args.save_path}clp_df_summary.csv")

        logging.info(f'the threshold{args.u} clean_loss:{clean_test_loss_avg_over_batch} bd_loss:{bd_test_loss_avg_over_batch} clean_acc:{test_acc} asr:{test_asr} ra:{test_ra}')


        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    clp.add_arguments(parser)
    args = parser.parse_args()
    method = clp(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = method.defense(args.result_file)