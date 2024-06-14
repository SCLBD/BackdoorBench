'''
This is the official implementation of the paper "Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features".
Paper link: https://openreview.net/forum?id=VFhN15Vlkj

@inproceedings{
zhu2023neural,
title={Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features},
author={Mingli Zhu and Shaokui Wei and Hongyuan Zha and Baoyuan Wu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=VFhN15Vlkj}}

This code provides the implementation of the NPD defense.
After training, the "neural polarizer layer will be saved separately as name "NP_layer.pt".
To evaluate the performance of NPD, please use the "evaluate.py" in the "utils/defense_utils/npd" folder.

Notations: There are some important hyper-parameters you can tune in the NPD defense.
    --target_layer_name: the selected layer to insert the polarizer
    --trigger_norm: the norm bound for the perturbation
    --norm_type: the norm type of the bound
    --inner_steps: the step for generate adversarial examples (relatively insensitive)
    --model_name: decide which polarizer structure to use (for ablation study)
    --lmd1|lmd2|lmd3|lmd4: hyperparameters of four different losses
    --lr: learning rate for learning the polarizer


basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor model
4. npd defense (train a neural polarizer layer):
    a. warm up with a small learning rate
    b. define optimizer
    c. preparation
    d. for each epoch of training the plug layer
        i. random targeted AE
        ii. training
5. test the result and get ASR, ACC, RC



'''

import argparse
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
from utils.trainer_cls import Metric_Aggregator
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index,choose_by_class
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.transforms as transforms
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization
from utils.defense_utils.npd.utils import disable_running_stats, enable_running_stats,AverageMeter

def output_feature_hook(module, input_, output_): ## sum pooling over spatial dimensions -> bs x c
    activation = None
    global out_feature_vector
    global input_feature_vector
    # access the layer output and convert it to a feature vector
    input_feature_vector = input_[0]
    out_feature_vector = output_
    if activation is not None:
        out_feature_vector = activation(out_feature_vector)
    if out_feature_vector.dim() > 2:
        out_feature_vector = torch.sum(torch.flatten(out_feature_vector, 2), 2) 
    else:
        out_feature_vector = out_feature_vector
    return None

def input_feature_hook(module, input_):
    if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
        modified_input = plug_layer(input_[0])
    elif args.model_name in ['linear', 'linear_light']:
        modified_input = input_[0] * plug_layer[0] + plug_layer[1]
    elif args.model_name == 'mlp':
        modified_input = nn.ReLU()(input_[0] * plug_layer[0] + plug_layer[1]) * plug_layer[2] + plug_layer[3]
    if args.use_residual == "yes":
        modified_input += input_[0]
    return modified_input

class NPD(defense):

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
    
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/npd/config.yaml", help='the path of yaml')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--print_freq', type=int, help='index of clean data')

        #set the parameter for the ft defense
        parser.add_argument("--ratio", type=float, help="ratio of clean samples, used for mix_dataset and legend")
        parser.add_argument("--lr", type=float, help="lr for defense")
        parser.add_argument("--epochs",type=int, help="epochs for defense")
        parser.add_argument("--target_layer_name", type=str, help="target layer of the model to insert the polarizer")
        parser.add_argument("--target_lab", default=None, type=int, help="the given target label of the attack")
        parser.add_argument("--trigger_norm", type=float, help="the norm bound of the perturbation")
        parser.add_argument("--norm_type", default="L2", type=str,choices=["L_inf","L2","L1"], help="the norm type of the bound")
        parser.add_argument("--inner_steps", type=int,help="the step for generate adversarial examples")
        parser.add_argument("--warm_epochs",type=int,help="warm up epochs for defense")
        parser.add_argument("--model_name", type=str, help="decide which polarizer structure to use")
        parser.add_argument("--lmd1",type=float, help="hyperparameters of loss1")
        parser.add_argument("--lmd2",type=float, help="hyperparameters of loss2")
        parser.add_argument("--lmd3",type=float, help="hyperparameters of loss3")
        parser.add_argument("--lmd4",type=float, help="hyperparameters of loss4")
        parser.add_argument("--max_init", action='store_true', default=False, help="the norm of the bound")
        parser.add_argument("--use_residual", default="no", type=str,choices=["yes","no"], help="use residual for the polarizer layer or not")   
        parser.add_argument("--target_class", default=0, type=int, help="target class of attacks, for evaluation only") 
         

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/npd/'
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
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def evaluation(self,args,model,clean_test_dataloader,data_bd_loader):
        # Load model
        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            plug_layer.eval()
        model.eval()
        model.to(args.device)
        # Create dataset
        h_in = args.target_layer.register_forward_pre_hook(input_feature_hook)
        target_class = args.target_class
        criterion = torch.nn.CrossEntropyLoss()
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        target_correct, target_total = 0, 0
        for i, (inputs, labels) in enumerate(clean_test_dataloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            target_correct += torch.sum((torch.argmax(outputs[:], dim=1) == target_class)*(labels[:] == target_class))
            target_total += torch.sum(labels[:] == target_class)

            total_clean_test += inputs.shape[0]
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
        # logging.info('Test Acc: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
        # logging.info('Test Acc (Target only): {:.3f}%({}/{})'.format(target_correct/target_total*100.0, target_correct, target_total))
        clean_acc = avg_acc_clean
        target_acc = (target_correct/target_total*100.0).item()
        clean_loss = test_loss/len(clean_test_dataloader)
        # Create dataset
    
        target_class = args.target_class 
        total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
        target_correct, target_total = 0, 0
        total_ra = 0
        for i, (inputs, labels, *others) in enumerate(data_bd_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            true_y = others[-1].to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_ra += torch.sum(torch.argmax(outputs[:], dim=1) == true_y[:])
            total_clean_test += inputs.shape[0]
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        test_asr = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
        bd_loss = test_loss/len(data_bd_loader)
        # logging.info('Test ASR: {:.3f}%({}/{})'.format(avg_acc_clean, total_clean_correct_test, total_clean_test))
        avg_ra = float(total_ra.item() * 100.0 / total_clean_test)
        # logging.info('Test RA: {:.3f}%({}/{})'.format(avg_ra, total_ra, total_clean_test))
        h_in.remove()
        return clean_acc, target_acc, test_asr, avg_ra, clean_loss, bd_loss

    def train_npd(self, model,plug_layer, train_dataloader,
                                   clean_test_dataloader,
                                   data_bd_loader,
                                   total_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   prefetch,
                                   prefetch_transform_attr_name,
                                   non_blocking,
                                   ):
        model.eval()
        target_layer = args.target_layer
        from tqdm import trange
        N_epochs = trange(args.epochs)
        lmd1, lmd2, lmd3, lmd4 = args.lmd1, args.lmd2, args.lmd3, args.lmd4
        bce_loss = True if lmd3 != 0 else False
        asr_loss = True if lmd4 != 0 else False
        df = None

        ## step1 warm up with a small learning rate
        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            opt_warm = torch.optim.SGD(plug_layer.parameters(), lr=0.0001)
        else:
            opt_warm = torch.optim.SGD(plug_layer, lr=0.0001)
        for epoch in range(args.warm_epochs):
            if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
                plug_layer.train()
            h_in = target_layer.register_forward_pre_hook(input_feature_hook)
            for batch_idx, (batch_x, batch_y,*other) in enumerate(train_dataloader):
                images = batch_x.to(args.device)
                labels = batch_y.to(args.device) 
                logits_1 = model(images)
                loss = criterion(logits_1, labels)
                opt_warm.zero_grad()
                loss.backward()
                opt_warm.step()
            h_in.remove()
            clean_acc, target_acc, test_asr, test_ra, clean_loss, bd_loss = self.evaluation(args,model,clean_test_dataloader,data_bd_loader)
            logging.info(f"Warm up epoch {epoch}, clean_acc, target_acc, test_asr, test_ra:{clean_acc}, {target_acc}, {test_asr}, {test_ra}" )
        logging.info(f"Warm up end, epochs: {args.warm_epochs},")
        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            plug_layer.eval()
        
        ## step2. define optimizer
        lr = args.lr
        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            plug_layer.to(args.device)
            opt_train = torch.optim.SGD(plug_layer.parameters(), lr=lr,weight_decay=5e-4,momentum=0.9)
            for param in plug_layer.parameters():
                param.retain_grad()
        elif args.model_name in ['linear','linear_light']:  
            opt_train = torch.optim.SGD([plug_layer[0],plug_layer[1]], lr=lr,weight_decay=5e-4,momentum=0.9)
        elif args.model_name == 'mlp':
            opt_train = torch.optim.SGD([plug_layer[0],plug_layer[1],plug_layer[2],plug_layer[3]], lr=lr,weight_decay=5e-4,momentum=0.9)
        
        ## step3. preparation
        loss_bound = 80
        args.rand_init = False
        if args.norm_type == 'L_inf':
            args.trigger_norm = 0.2
            args.adv_lr = args.trigger_norm*2.2
            inner_steps = 10
            args.max_init = True
        elif args.norm_type == 'L2':
            args.adv_lr = 0.1
            inner_steps = args.inner_steps

        ## step4 train the plug layer
        agg = Metric_Aggregator()
        for epoch in N_epochs:
            
            losses = AverageMeter()
            top1 = AverageMeter()

            h_in = target_layer.register_forward_pre_hook(input_feature_hook)
            for batch_idx, (batch_x, batch_y, ori_idx, poi_indicator, batch_true_y) in enumerate(train_dataloader):
                if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
                    plug_layer.eval()
                images = batch_x.to(args.device)
                labels = batch_y.to(args.device) 
                bsz = batch_y.shape[0]
                batch_true_y = batch_true_y.to(args.device)
                logits_original = model.forward(images)
                ori_lab = torch.argmax(logits_original, axis=1).long()

                # 4.1 random targeted AE
                
                if args.rand_init:
                    batch_pert = torch.rand(size=[batch_x.shape[0], args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
                    batch_pert.data = batch_pert.data * 2 * args.trigger_norm - args.trigger_norm
                    batch_pert.data = self.projection(batch_pert.data, args)
                elif args.max_init:
                    batch_pert = torch.zeros([batch_x.shape[0], args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)
                    batch_pert.data += args.trigger_norm
                else:
                    batch_pert = torch.zeros([batch_x.shape[0], args.input_channel, args.input_height, args.input_width], requires_grad=True, device=args.device)

                batch_opt = torch.optim.SGD(params=[batch_pert], lr=args.adv_lr)
                logits_original[torch.arange(len(batch_true_y)), batch_true_y] = -10
                target_lab = torch.argmax(logits_original, axis=1).long()
                # if batch_idx == 0:
                #     logging.info(f'epoch: {epoch} batch_idx 0, target_lab: {target_lab}')

                for _ in range(inner_steps):
                    
                    pert_images = self.get_perturbed_image(images, batch_pert)
                    per_logits = model.forward(pert_images)
                    loss = F.cross_entropy(per_logits, target_lab, reduction='none') 
                    loss_regu = torch.mean(loss.clamp_(-loss_bound,loss_bound))
                    batch_opt.zero_grad()
                    loss_regu.backward()
                    if args.dataset == 'tiny':
                        batch_opt.step() 
                    else:
                        batch_pert.data = batch_pert.data - args.adv_lr * torch.sign(batch_pert.grad.data) # FGSM
                    batch_pert = self.projection(batch_pert, args)
                    with torch.no_grad():
                        per_logits = model.forward(self.get_perturbed_image(images, batch_pert))
                        acc = (per_logits.argmax(dim=1) == target_lab).sum().item()/len(batch_true_y)
                    # logging.info(f'!!!!!!AT Predicted acc: {acc}')
                    if acc > 0.6:
                        break
                # logging.info(f'pert norm: {torch.mean(torch.sum(batch_pert ** 2, dim=(1, 2, 3)) ** 0.5)}')
                
                pert = batch_pert.detach()
                batch_pert_imgs = self.get_perturbed_image(images, pert).detach()

                # 4.2 training
                for _ in range(1):
                    opt_train.zero_grad()
                    if args.dataset == 'tiny':
                        plug_layer.train()
                        enable_running_stats(plug_layer)
                    # loss1
                    logits_1 = model(images)
                    loss_1 = criterion(logits_1, labels)
                    acc_1 = (logits_1.argmax(dim=1) == labels).sum().item()/len(labels)   
                    if args.dataset == 'tiny': 
                        disable_running_stats(plug_layer)    
                    loss, loss_2,loss_3,loss_4 = torch.tensor(0.0).to(args.device), torch.tensor(0.0).to(args.device), torch.tensor(0.0).to(args.device), torch.tensor(0.0).to(args.device)
                    acc_2 = 0
                    logits_adv = model.forward(batch_pert_imgs)
                    # loss_2              
                    loss_2 += criterion(logits_adv, ori_lab.detach())
                    acc_2 += (logits_adv.argmax(dim=1) == batch_true_y).sum().item()/len(batch_true_y)
                    ## loss3  
                    adv_probs = F.softmax(logits_adv, dim=1)
                    tmp2 = torch.argsort(adv_probs, dim=1)[:, -2:]
                    if bce_loss:  
                        new_y = torch.where(tmp2[:, -1] == labels, tmp2[:, -2], tmp2[:, -1])  
                        loss_3 += torch.mean(F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction='none'))
                    if asr_loss:  
                        loss_4 += torch.mean(F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), target_lab, reduction='none'))
                    loss = lmd1 * loss_1 + lmd2 * loss_2 + lmd3 * loss_3 + lmd4 * loss_4
                    loss.backward() 
                    opt_train.step()
                
                losses.update(loss.item(), bsz)
                top1.update(acc_1, bsz)

            h_in.remove()

            clean_acc, target_acc, test_asr, test_ra, clean_loss, bd_loss = self.evaluation(args,model,clean_test_dataloader,data_bd_loader)
    
            agg(
                {
                    "train_epoch_loss_avg_over_batch": losses.avg,
                    "train_acc": top1.avg,
                    "clean_test_loss_avg_over_batch": clean_loss,
                    "bd_test_loss_avg_over_batch" : bd_loss,
                    "test_acc" : clean_acc,
                    "test_asr" : test_asr,
                    "test_ra" : test_ra,
                }
            )
            agg.to_dataframe().to_csv(f"{args.save_path}npd_df.csv")

        agg.summary().to_csv(f"{args.save_path}npd_df_summary.csv")
    
        return plug_layer

    def prepare_polarizer(self,model, args,data_loader):
        assert args.model == "preactresnet18", "Only support preactresnet18"
        for param in model.parameters():
            param.requires_grad = False
        logging.info(f"target_layer_name: {args.target_layer_name}",)
        module_dict = dict(model.named_modules())
        assert args.target_layer_name in module_dict.keys()
        target_layer = module_dict[args.target_layer_name]
        args.target_layer = target_layer
        random_batch, *other_info = next(iter(data_loader))
        random_batch = random_batch.to(args.device)
        # Collect random feature vector
        h_out = target_layer.register_forward_hook(output_feature_hook)
        model(random_batch)
        base_feature_clean = out_feature_vector.detach().cpu().numpy()
        base_clean_input = input_feature_vector.detach().cpu().numpy()
        h_out.remove()

        size = base_clean_input.shape
        # change the first element of tuple to 1
        size = list(size)
        use_bias = False
        chan = size[1]
        kernel_size = 1

        global plug_layer
        if args.model_name == 'mlp':
            weight_size = [1,size[1],size[2],size[3]]
            w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
            b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
            relu = nn.ReLU()
            w2 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
            b2 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
            plug_layer = [w1,b1,w2,b2]
        elif args.model_name == 'onlyconv':
            plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=kernel_size, stride=1, padding=0, bias=use_bias))
            plug_layer[0].weight.data.fill_(0)
            for i in range(chan):
                plug_layer[0].weight.data[i, i, 0, 0] = 1
        elif args.model_name == 'convbn' or args.model_name == 'lightconv':
            plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=kernel_size, stride=1, padding=0, bias=use_bias), nn.BatchNorm2d(chan))
            plug_layer[0].weight.data.fill_(0)
            for i in range(chan):
                plug_layer[0].weight.data[i, i, 0, 0] = 1
        elif args.model_name == 'linear':
            weight_size = [1,size[1],size[2],size[3]]
            w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
            b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
            plug_layer = [w1,b1]
        elif args.model_name == 'linear_light':
            weight_size = [1,size[1],1,1]
            w1 = torch.ones(size = weight_size, requires_grad=True, device= args.device)
            b1 = torch.zeros(size = weight_size, requires_grad=True, device= args.device)
            plug_layer = [w1,b1]
        elif args.model_name == 'twoconv':
            plug_layer = nn.Sequential(nn.Conv2d(chan, chan, kernel_size=kernel_size, stride=1, padding=0, bias=use_bias), nn.BatchNorm2d(chan), nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0, bias=use_bias),nn.BatchNorm2d(chan))
            plug_layer[0].weight.data.fill_(0)
            for i in range(chan):
                plug_layer[0].weight.data[i, i, 0, 0] = 1
            plug_layer[2].weight.data.fill_(0)
            for i in range(chan):
                plug_layer[2].weight.data[i, i, 0, 0] = 1
        else:
            raise NotImplementedError

        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            plug_layer.to(args.device)
            for param in plug_layer.parameters():
                param.retain_grad()
        else:
            for param in plug_layer:
                param.to(args.device)
                param.retain_grad()
        return plug_layer

    def get_perturbed_image(self, images, pert): # batch_x 与pert得到normalize之后的图像
        images_wo_trans = self.denormalization(images) + pert
        images_wo_trans = images_wo_trans.clamp(0, 1)
        images_with_trans = self.normalization(images_wo_trans)
        return images_with_trans

    def projection(self, pert, args): 
        if args.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -args.trigger_norm , args.trigger_norm)
        elif args.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > args.trigger_norm:
                    pert.data[i] = pert.data[i] * args.trigger_norm / norm[i].item()
        elif args.norm_type == 'L2':
            norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
            for i in range(pert.shape[0]):
                if norm[i] > args.trigger_norm:
                    pert.data[i] = pert.data[i] * args.trigger_norm / norm[i].item()
        elif args.norm_type == 'Reg':
            pass
        else:
            raise NotImplementedError
        return pert

    def mitigation(self):
        args=self.args
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        if hasattr(args,"checkpoint_path") and args.checkpoint_path != None:
            file_path = 'record/' + args.checkpoint_path 
            checkpoint_path = load_attack_result(file_path + '/defense_result.pt')
            model.load_state_dict(checkpoint_path['model'])
        else:
            model.load_state_dict(self.result['model'])

        if "," in self.args.device:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
        else:
            model.to(self.args.device)
        model.eval()
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        criterion = argparser_criterion(args)

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False) # train = False for better performance
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        # ran_idx = choose_index(self.args, data_all_length) 
        ran_idx = choose_by_class(args,clean_dataset) # choose by class for more fair comparison
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        for trans_t in train_tran.transforms:
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
        self.normalization = trans_t
        self.denormalization = denormalizer

        ## 1. prepare plug layer
        plug_layer = self.prepare_polarizer(model, args, trainloader)
        
        ## 2. train plug layer
        plug_layer = self.train_npd(
            model,
            plug_layer,
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='da',
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        ## 3. save the plug layer
        if args.model_name in ['twoconv','lightconv','onlyconv','convbn']:
            model_save = {
            # 'model_state_dict': model.state_dict(),
            'target_layer': args.target_layer_name,
            'plug_name': args.model_name,
            'plug_model':plug_layer.state_dict(),
            }
        else:
            model_save = {
            # 'model_state_dict': model.state_dict(),
            'target_layer':  args.target_layer_name,
            'plug_name': args.model_name,
            'plug_model':plug_layer,
            }

        torch.save(model_save, args.save_path + f"/plug_layer.pt")

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
    NPD.add_arguments(parser)
    args = parser.parse_args()
    ft_method = NPD(args)
    result = ft_method.defense(args.result_file)