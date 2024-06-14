'''
Anti-backdoor learning: Training clean models on poisoned data.
This file is modified based on the following source:
link : https://github.com/bboylyg/ABL.
The defense method is called abl.
@article{li2021anti,
            title={Anti-backdoor learning: Training clean models on poisoned data},
            author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={14900--14912},
            year={2021}
            }
The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

class LGALoss(nn.Module):
    def __init__(self, gamma, criterion):
        super(LGALoss, self).__init__()
        self.gamma = gamma
        self.criterion = criterion
        return
    
    def forward(self,output,target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent

class FloodingLoss(nn.Module):
    def __init__(self, flooding, criterion):
        super(FloodingLoss, self).__init__()
        self.flooding = flooding
        self.criterion = criterion
        return
    
    def forward(self,output,target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = (loss - self.flooding).abs() + self.flooding
        return loss_ascent


def adjust_learning_rate(optimizer, epoch, args):
    '''set learning rate during the process of pretraining model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.tuning_epochs:
        lr = args.lr
    else:
        lr = 0.01
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_loss_value(args, poisoned_data, model_ascent):
    '''Calculate loss value per example
    args:
        Contains default parameters
    poisoned_data:
        the train dataset which contains backdoor data
    model_ascent:
        the model after the process of pretrain
    '''
    # Define loss function
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target,_,_,_) in tqdm(enumerate(example_data_loader, start=0)):
        
        img = img.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    logging.info(f'Top ten loss value: {losses_record_arr[losses_idx[:10]]}')

    return losses_idx

def isolate_data(args, result, losses_idx):
    '''isolate the backdoor data with the calculated loss
    args:
        Contains default parameters
    result:
        the attack result contain the train dataset which contains backdoor data
    losses_idx:
        the index of order about the loss value for each data 
    '''
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = args.isolation_ratio
    perm = losses_idx[0: int(len(losses_idx) * ratio)]
    permnot = losses_idx[int(len(losses_idx) * ratio):]
    tf_compose = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    train_dataset = result['bd_train'].wrapped_dataset
    data_set_without_tran = train_dataset
    data_set_isolate = result['bd_train']
    data_set_isolate.wrapped_dataset = data_set_without_tran
    data_set_isolate.wrap_img_transform = tf_compose

    data_set_other_without_tran = data_set_without_tran.copy()
    data_set_other = dataset_wrapper_with_transform(
            data_set_other_without_tran,
            tf_compose,
            None,
        )
    # x = result['bd_train']['x']
    # y = result['bd_train']['y']

    data_set_isolate.subset(perm)
    data_set_other.subset(permnot)

    # isolation_examples = list(zip([x[ii] for ii in perm],[y[ii] for ii in perm]))
    # other_examples = list(zip([x[ii] for ii in permnot],[y[ii] for ii in permnot]))
    
    logging.info('Finish collecting {} isolation examples: '.format(len(data_set_isolate)))
    logging.info('Finish collecting {} other examples: '.format(len(data_set_other)))

    return data_set_isolate, data_set_other



def learning_rate_finetuning(optimizer, epoch, args):
    '''set learning rate during the process of finetuing model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_unlearning(optimizer, epoch, args):
    '''set learning rate during the process of unlearning model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class abl(defense):
    r"""Anti-backdoor learning: Training clean models on poisoned data.
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        abl.add_arguments(parser)
        args = parser.parse_args()
        abl_method = abl(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = abl_method.defense(args.result_file)
    
    .. Note::
        @article{li2021anti,
            title={Anti-backdoor learning: Training clean models on poisoned data},
            author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={14900--14912},
            year={2021}
            }

    Args:
        baisc args: in the base class
        tuning_epochs (int): number of the first tuning epochs to run
        finetuning_ascent_model (bool): whether finetuning model after sperate the poisoned data
        finetuning_epochs (int): number of the finetuning epochs to run
        unlearning_epochs (int): number of the unlearning epochs to run
        lr_finetuning_init (float): initial finetuning learning rate
        lr_unlearning_init (float): initial unlearning learning rate
        momentum (float): momentum of sgd during the process of finetuning and unlearning
        weight_decay (float): weight decay of sgd during the process of finetuning and unlearning
        isolation_ratio (float): ratio of isolation data from the whole poisoned data
        gradient_ascent_type (str): type of gradient ascent (LGA, Flooding)
        gamma (float): value of gamma for LGA
        flooding (float): value of flooding for Flooding
        
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
        parser.add_argument('--interval', type=int, help='frequency of save model')
    
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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/abl/config.yaml", help='the path of yaml')

        #set the parameter for the abl defense
        parser.add_argument('--tuning_epochs', type=int, help='number of tune epochs to run')
        parser.add_argument('--finetuning_ascent_model', type=bool, help='whether finetuning model')
        parser.add_argument('--finetuning_epochs', type=int, help='number of finetuning epochs to run')
        parser.add_argument('--unlearning_epochs', type=int, help='number of unlearning epochs to run')
        parser.add_argument('--lr_finetuning_init', type=float, help='initial finetuning learning rate')
        parser.add_argument('--lr_unlearning_init', type=float, help='initial unlearning learning rate')
        parser.add_argument('--momentum', type=float, help='momentum')
        parser.add_argument('--weight_decay', type=float, help='weight decay')
        parser.add_argument('--isolation_ratio', type=float, help='ratio of isolation data')
        parser.add_argument('--gradient_ascent_type', type=str, help='type of gradient ascent')
        parser.add_argument('--gamma', type=float, help='value of gamma')
        parser.add_argument('--flooding', type=float, help='value of flooding')
        

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/{self.__class__.__name__}/'
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
        result = self.result 
        ###a. pre-train model
        poisoned_data, model_ascent = self.pre_train(args,result)
        
        ###b. isolate the special data(loss is low) as backdoor data
        losses_idx = compute_loss_value(args, poisoned_data, model_ascent)
        logging.info('----------- Collect isolation data -----------')
        isolation_examples, other_examples = isolate_data(args, result, losses_idx)

        ###c. unlearn the backdoor data and learn the remaining data
        model_new = self.train_unlearning(args,result,model_ascent,isolation_examples,other_examples)

        result = {}
        result['model'] = model_new
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_new.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

    def pre_train(self, args, result):
        '''Pretrain the model with raw data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        '''
        agg = Metric_Aggregator()
        # Load models
        logging.info('----------- Network Initialization --------------')
        model_ascent = generate_cls_model(args.model,args.num_classes)
        if "," in self.device:
            model_ascent = torch.nn.DataParallel(
                model_ascent,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model_ascent.device_ids[0]}'
            model_ascent.to(self.args.device)
        else:
            model_ascent.to(self.args.device)
        logging.info('finished model init...')
        # initialize optimizer 
        # because the optimizer has parameter nesterov
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        # define loss functions
        # recommend to use cross entropy
        criterion = argparser_criterion(args).to(args.device)
        if args.gradient_ascent_type == 'LGA':
            criterion = LGALoss(args.gamma,criterion).to(args.device)
        elif args.gradient_ascent_type == 'Flooding':
            criterion = FloodingLoss(args.flooding,criterion).to(args.device)
        else:
            raise NotImplementedError

        logging.info('----------- Data Initialization --------------')

        # tf_compose = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        tf_compose = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        train_dataset = result['bd_train'].wrapped_dataset
        data_set_without_tran = train_dataset
        data_set_o = result['bd_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = tf_compose
        
        # data_set_isolate = result['bd_train']
        # data_set_isolate.wrapped_dataset = data_set_without_tran
        # data_set_isolate.wrap_img_transform = tf_compose

        # # data_set_other = copy.deepcopy(data_set_isolate)
        # # x = result['bd_train']['x']
        # # y = result['bd_train']['y']
        # losses_idx = range(50000)
        # ratio = args.isolation_ratio
        # perm = losses_idx[0: int(len(losses_idx) * ratio)]
        # permnot = losses_idx[int(len(losses_idx) * ratio):]
        # data_set_isolate.subset(perm)
        # data_set_o.subset(permnot)
        # data_set_other = copy.deepcopy(data_set_o)
        poisoned_data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        logging.info('----------- Train Initialization --------------')
        for epoch in range(0, args.tuning_epochs):
            logging.info("Epoch {}:".format(epoch + 1))
            adjust_learning_rate(optimizer, epoch, args)
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra = self.train_step(args, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model_ascent,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}pre_train_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}pre_train_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}pre_train_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "model": model_ascent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.checkpoint_save + "pre_train_state_dict.pt")

        agg.summary().to_csv(f"{args.save_path}pre_train_df_summary.csv")

        return data_set_o, model_ascent

    def train_unlearning(self, args, result, model_ascent, isolate_poisoned_data, isolate_other_data):
        '''train the model with remaining data and unlearn the backdoor data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        model_ascent:
            the model after pretrain
        isolate_poisoned_data:
            the dataset of 'backdoor' data
        isolate_other_data:
            the dataset of remaining data
        '''
        agg = Metric_Aggregator()
        # Load models
        ### TODO: load model from checkpoint
        # logging.info('----------- Network Initialization --------------')
        # if "," in args.device:
        #     model_ascent = torch.nn.DataParallel(
        #         model_ascent,
        #         device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        #     )
        # else:
        #     model_ascent.to(args.device)
        # model_ascent.to(args.device)
        logging.info('Finish loading ascent model...')
        # initialize optimizer
        # Because nesterov we do not use other optimizer
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        # define loss functions
        # you can use other criterion, but the paper use cross validation to unlearn sample
        if args.device == 'cuda':
            criterion = argparser_criterion(args).cuda()
        else:
            criterion = argparser_criterion(args)
        
        tf_compose_finetuning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
        tf_compose_unlearning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    
        isolate_poisoned_data.wrap_img_transform = tf_compose_finetuning
        isolate_poisoned_data_loader = torch.utils.data.DataLoader(dataset=isolate_poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        )

        isolate_other_data.wrap_img_transform = tf_compose_unlearning
        isolate_other_data_loader = torch.utils.data.DataLoader(dataset=isolate_other_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                )

        test_tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        data_bd_testset = result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        logging.info('----------- Train Initialization --------------')

        if args.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            logging.info('----------- Finetuning isolation model --------------')
            for epoch in range(0, args.finetuning_epochs):
                learning_rate_finetuning(optimizer, epoch, args)
                train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra = self.train_step(args, isolate_other_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

                clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra = self.eval_step(
                    model_ascent,
                    data_clean_loader,
                    data_bd_loader,
                    args,
                )

                agg({
                    "epoch": epoch,

                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,
                    "train_acc_clean_only": train_clean_acc,
                    "train_asr_bd_only": train_asr,
                    "train_ra_bd_only": train_ra,

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })

                train_loss_list.append(train_epoch_loss_avg_over_batch)
                train_mix_acc_list.append(train_mix_acc)
                train_clean_acc_list.append(train_clean_acc)
                train_asr_list.append(train_asr)
                train_ra_list.append(train_ra)

                clean_test_loss_list.append(clean_test_loss_avg_over_batch)
                bd_test_loss_list.append(bd_test_loss_avg_over_batch)
                ra_test_loss_list.append(ra_test_loss_avg_over_batch)
                test_acc_list.append(test_acc)
                test_asr_list.append(test_asr)
                test_ra_list.append(test_ra)

                general_plot_for_epoch(
                    {
                        "Train Acc": train_mix_acc_list,
                        "Test C-Acc": test_acc_list,
                        "Test ASR": test_asr_list,
                        "Test RA": test_ra_list,
                    },
                    save_path=f"{args.save_path}finetune_acc_like_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "Train Loss": train_loss_list,
                        "Test Clean Loss": clean_test_loss_list,
                        "Test Backdoor Loss": bd_test_loss_list,
                        "Test RA Loss": ra_test_loss_list,
                    },
                    save_path=f"{args.save_path}finetune_loss_metric_plots.png",
                    ylabel="percentage",
                )

                agg.to_dataframe().to_csv(f"{args.save_path}finetune_df.csv")

                if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                    state_dict = {
                        "model": model_ascent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch_current": epoch,
                    }
                    torch.save(state_dict, args.checkpoint_save + "finetune_state_dict.pt")
        agg.summary().to_csv(f"{args.save_path}finetune_df_summary.csv")


        best_acc = 0
        best_asr = 0
        logging.info('----------- Model unlearning --------------')
        for epoch in range(0, args.unlearning_epochs):
            
            learning_rate_unlearning(optimizer, epoch, args)
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra = self.train_step_unlearn(args, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model_ascent,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}unlearn_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}unlearn_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}unlearn_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "model": model_ascent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.checkpoint_save + "unlearn_state_dict.pt")
        
        agg.summary().to_csv(f"{args.save_path}unlearn_df_summary.csv")
        agg.summary().to_csv(f"{args.save_path}abl_df_summary.csv")
        return model_ascent

    
    def train_step(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        '''Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        '''
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        model_ascent.train()

        for idx, (img, target, original_index, poison_indicator, original_targets) in enumerate(train_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            pred = model_ascent(img)
            loss_ascent = criterion(pred,target)

            losses += loss_ascent * img.size(0)
            size += img.size(0)
            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(pred, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_original_index_list.append(original_index.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra

    def train_step_unlearn(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        '''Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        '''
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        model_ascent.train()

        for idx, (img, target, original_index, poison_indicator, original_targets) in enumerate(train_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            pred = model_ascent(img)
            loss_ascent = criterion(pred,target)

            losses += loss_ascent * img.size(0)
            size += img.size(0)
            optimizer.zero_grad()
            (-loss_ascent).backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(pred, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_original_index_list.append(original_index.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra

    def eval_step(
            self,
            netC,
            clean_test_dataloader,
            bd_test_dataloader,
            args,
    ):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    abl.add_arguments(parser)
    args = parser.parse_args()
    abl_method = abl(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    result = abl_method.defense(args.result_file)