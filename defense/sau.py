'''
This is the official implementation of the paper "Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples" (https://arxiv.org/pdf/2307.10562.pdf) in PyTorch.
Implementation by: Shaokui Wei (the first author of the paper)

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. sau defense:
        a. get some clean data
        b. SAU:
            b.1 generate the shared adversarial examples
            b.2 unlearn the backdoor model by the pertubation
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
import random
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch, given_dataloader_test
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization, get_dataset_denormalization
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

from itertools import repeat
import torchvision.transforms as transforms

from torch import Tensor


class Shared_PGD():
    def __init__(self, model, model_ref, beta_1 = 0.01, beta_2 = 1, norm_bound = 0.2, norm_type = 'L_inf', step_size = 0.2, num_steps = 5, init_type = 'max', loss_func = torch.nn.CrossEntropyLoss(), pert_func = None, verbose = False):
        '''
        PGD attack for generating shared adversarial examples. 
        See "Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples" (https://arxiv.org/pdf/2307.10562.pdf) for more details.
        Implemented by Shaokui Wei (the first author of the paper) in PyTorch.
        The code is originally implemented as a part of BackdoorBench but is not dependent on BackdoorBench, and can be used independently.
        
        args:
            model: the model to be attacked
            model_ref: the reference model to be attacked
            beta_1: the weight of adversarial loss, e.g. 0.01
            beta_2: the weight of shared loss, e.g. 1
            norm_bound: the bound of the norm of perturbation, e.g. 0.2
            norm_type: the type of norm, choose from ['L_inf', 'L1', 'L2', 'Reg']
            step_size: the step size of PGD, e.g. 0.2
            num_steps: the number of steps of PGD, e.g. 5
            init_type: the type of initialization of perturbation, choose from ['zero', 'random', 'max', 'min']
            loss_func: the loss function, e.g. nn.CrossEntropyLoss()
            pert_func: the function to process the perturbation and image, e.g. add the perturbation to image
            verbose: whether to print the information of the attack
        '''

        self.model = model
        self.model_ref = model_ref
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.norm_bound = norm_bound
        self.norm_type = norm_type
        self.step_size = step_size
        self.num_steps = num_steps
        self.init_type = init_type
        self.loss_func = loss_func
        self.verbose = verbose

        if pert_func is None:
            # simply add x to perturbation
            self.pert_func = lambda x, pert: x + pert
        else:
            self.pert_func = pert_func
            
    def projection(self, pert):
        if self.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -self.norm_bound , self.norm_bound)
        elif self.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'L2':
            norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'Reg':
            pass
        else:
            raise NotImplementedError
        return pert
    
    def init_pert(self, batch_pert):
        if self.init_type=='zero':
            batch_pert.data = batch_pert.data*0
        elif self.init_type=='random':
            batch_pert.data = torch.rand_like(batch_pert.data)
        elif self.init_type=='max':
            batch_pert.data = batch_pert.data + self.norm_bound
        elif self.init_type=='min':
            batch_pert.data = batch_pert.data - self.norm_bound
        else:
            raise NotImplementedError

        return self.projection(batch_pert)

    def attack(self, images, labels, max_eps = 1, min_eps = 0):
        # Set max_eps and min_eps to valid range

        model = self.model
        model_ref = self.model_ref

        batch_pert = torch.zeros_like(images, requires_grad=True)
        batch_pert = self.init_pert(batch_pert)

        for _ in range(self.num_steps):   
            pert_image = self.pert_func(images, batch_pert)
            ori_lab = torch.argmax(model.forward(images),axis = 1).long()
            ori_lab_ref = torch.argmax(model_ref.forward(images),axis = 1).long()

            per_logits = model.forward(pert_image)
            per_logits_ref = model_ref.forward(pert_image)

            pert_label = torch.argmax(per_logits, dim=1)
            pert_label_ref = torch.argmax(per_logits_ref, dim=1)
                
            success_attack = pert_label != ori_lab
            success_attack_ref = pert_label_ref != ori_lab_ref
            common_attack = torch.logical_and(success_attack, success_attack_ref)
            shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

            # Adversarial loss
            # use early stop or loss clamp to avoid very large loss
            loss_adv = torch.tensor(0.0).to(images.device)
            if torch.logical_not(success_attack).sum()!=0:
                loss_adv += F.cross_entropy(per_logits, labels, reduction='none')[torch.logical_not(success_attack)].sum()
            if torch.logical_not(success_attack_ref).sum()!=0:
                loss_adv += F.cross_entropy(per_logits_ref, labels, reduction='none')[torch.logical_not(success_attack_ref)].sum()
            loss_adv = - loss_adv/2/images.shape[0]

            # Shared loss
            # JS divergence version (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
            p_model = F.softmax(per_logits, dim=1).clamp(min=1e-8)
            p_ref = F.softmax(per_logits_ref, dim=1).clamp(min=1e-8)
            mix_p = 0.5*(p_model+p_ref)
            loss_js = 0.5*(p_model*p_model.log() + p_ref*p_ref.log()) - 0.5*(p_model*mix_p.log() + p_ref*mix_p.log())
            loss_cross = loss_js[torch.logical_not(shared_attack)].sum(dim=1).sum()/images.shape[0]

            # Update pert              
            batch_pert.grad = None
            loss_ae = self.beta_1 * loss_adv + self.beta_2 * loss_cross
            loss_ae.backward()

            batch_pert.data = batch_pert.data - self.step_size * batch_pert.grad.sign()
    
            # Projection
            batch_pert = self.projection(batch_pert)

            # Optimal: projection to S and clip to [min_eps, max_eps] to ensure the perturbation is valid. It is not necessary for backdoor defense as done in i-BAU.
            # Mannually set the min_eps and max_eps to match the dataset normalization
            # batch_pert.data = torch.clamp(batch_pert.data, min_eps, max_eps)

            if torch.logical_not(shared_attack).sum()==0:
                break
        if self.verbose:
            print(f'Maximization End: \n Adv h: {success_attack.sum().item()}, Adv h_0: {success_attack_ref.sum().item()}, Adv Common: {common_attack.sum().item()}, Adv Share: {shared_attack.sum().item()}.\n Loss adv {loss_adv.item():.4f}, Loss share {loss_cross.item():.4f}, Loss total {loss_ae.item():.4f}.\n L1 norm: {torch.sum(batch_pert[0].abs().sum()):.4f}, L2 norm: {torch.norm(batch_pert[0]):.4f}, Linf norm: {torch.max(batch_pert[0].abs()):.4f}')                    

        return batch_pert.detach()

class sau(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/sau/config.yaml", help='the path of yaml')

        
        ###### sau defense parameter ######
        # defense setting
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        # hyper params
        parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')
        parser.add_argument('--n_rounds', type=int, help='the maximum number of unelarning rounds')
        
        ## Minimization part
        parser.add_argument('--outer_steps', default=1, type=int,
                            help='steps for outer loop, the number of unlearning rounds')
        parser.add_argument('--lmd_1', type=float,
                            help='clean acc, L_cl')
        parser.add_argument('--lmd_2', type=float,
                            help='AT acc. By default, lmd_2 = 0 and AT is not used.')
        parser.add_argument('--lmd_3', type=float,
                            help=' shared adv risk, L_sar')
        
        ## Maximization part
        parser.add_argument('--beta_1', type=float,
                            help='L_adv')
        parser.add_argument('--beta_2', type=float,
                            help='L_share')

        ### PGD setting        
        parser.add_argument('--trigger_norm', type=float,
                            help='threshold for PGD. Larger may not be good.')
        
        parser.add_argument('--pgd_init', type=str, 
                            help='init type for pgd. zero|random|max|min')

        parser.add_argument('--norm_type', type=str,
                            help='type of norm used for generating perturbation. L1|L2|L_inf|Reg')

        parser.add_argument('--adv_lr', type=float,
                            help='lr for pgd')

        parser.add_argument('--adv_steps', type=int,
                            help='number of steps for pgd')

        ## optimization setting
        parser.add_argument('--train_mode', action='store_true',
                            default=False, help='Fix BN parameters or not. Fixing BN leads to higher ACC but also higher ASR.')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/sau/'
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
    

    def mitigation(self):
        fix_random(self.args.random_seed)

        # initialize models
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])

        model_ref = generate_cls_model(self.args.model,self.args.num_classes)
        model_ref.load_state_dict(self.result['model'])
    
  
        if "," in self.args.device:
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in self.args.device[5:].split(",")])
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)

            model_ref = torch.nn.DataParallel(model_ref, device_ids=[int(i) for i in self.args.device[5:].split(",")])
            self.args.device = f'cuda:{model_ref.device_ids[0]}'
            model_ref.to(self.args.device)
        else:
            model.to(self.args.device)
            model_ref.to(self.args.device)

        outer_opt, scheduler = argparser_opt_scheduler(model, self.args)
        
        # a. get some clean data
        logging.info("Fetch some samples from clean train dataset.")

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)

        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length)
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')

        clean_dataset.subset(ran_idx)

        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        ## set testing dataset
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        # b. unlearn the backdoor model by the pertubation
        logging.info("=> Conducting Defence..")
        model.eval()
        model_ref.eval()

        clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    ra_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.eval_step(
                        model,
                        data_clean_loader,
                        data_bd_loader,
                        args,
                    )        

        logging.info('Initial State: clean test loss: {:.4f}, bd test loss: {:.4f}, ra test loss: {:.4f}, test acc: {:.4f}, test asr: {:.4f}, test ra: {:.4f}'.format(clean_test_loss_avg_over_batch, bd_test_loss_avg_over_batch, ra_test_loss_avg_over_batch, test_acc, test_asr, test_ra))


        normalization = get_dataset_normalization(args.dataset)
        denormalization = get_dataset_denormalization(normalization)


        
        def get_perturbed_image(images, pert, train = True):
            images_wo_trans = denormalization(images) + pert
            images_with_trans = normalization(images_wo_trans)
            return images_with_trans
        
        Shared_PGD_Attacker = Shared_PGD(model = model, 
                                         model_ref = model_ref, 
                                         beta_1 = args.beta_1, 
                                         beta_2 = args.beta_2, 
                                         norm_bound = args.trigger_norm, 
                                         norm_type = args.norm_type, 
                                         step_size = args.adv_lr, 
                                         num_steps = args.adv_steps, 
                                         init_type = args.pgd_init,
                                         loss_func = torch.nn.CrossEntropyLoss(), 
                                         pert_func = get_perturbed_image, 
                                         verbose = True)

        agg = Metric_Aggregator()
        for round in range(args.n_rounds):
            
            for images, labels, original_index, poison_indicator, original_targets in trainloader:
                images = images.to(args.device)
                labels = labels.to(args.device)

                max_eps = 1 - denormalization(images)
                min_eps = -denormalization(images)

                batch_pert = Shared_PGD_Attacker.attack(images, labels, max_eps, min_eps)

                for _ in range(args.outer_steps):
                    pert_image = get_perturbed_image(images, batch_pert.detach())

                    if args.train_mode:
                        model.train()

                    concat_images = torch.cat([images, pert_image], dim=0)
                    concat_logits = model.forward(concat_images)
                    logits, per_logits = torch.split(concat_logits, images.shape[0], dim=0)
                    model.eval()
                    
                    logits_ref = model_ref(images)
                    per_logits_ref = model_ref.forward(pert_image)

                    # Get prediction
                    ori_lab = torch.argmax(logits,axis = 1).long()
                    ori_lab_ref = torch.argmax(logits_ref,axis = 1).long()

                    pert_label = torch.argmax(per_logits, dim=1)
                    pert_label_ref = torch.argmax(per_logits_ref, dim=1)
                     
                    success_attack = pert_label != labels
                    success_attack_ref = pert_label_ref != labels
                    success_attack_ref = success_attack_ref & (pert_label_ref != ori_lab_ref)
                    common_attack = torch.logical_and(success_attack, success_attack_ref)
                    shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

                    # Clean loss
                    loss_cl = F.cross_entropy(logits, labels, reduction='mean')
                    
                    # AT loss
                    loss_at = F.cross_entropy(per_logits, labels, reduction='mean')
                    
                    
                    # Shared loss
                    potential_poison = success_attack_ref

                    if potential_poison.sum() == 0:
                        loss_shared = torch.tensor(0.0).to(args.device)
                    else:
                        one_hot = F.one_hot(pert_label_ref, num_classes=args.num_classes)
                        
                        neg_one_hot = 1 - one_hot
                        neg_p = (F.softmax(per_logits, dim = 1)*neg_one_hot).sum(dim = 1)[potential_poison]
                        pos_p = (F.softmax(per_logits, dim = 1)*one_hot).sum(dim = 1)[potential_poison]
                        
                        # clamp the too small values to avoid nan and discard samples with p<1% to be shared
                        # Note: The below equation combine two identical terms in math. Although they are the same in math, they are different in implementation due to the numerical issue. 
                        #       Combining them can reduce the numerical issue.

                        loss_shared = (-torch.sum(torch.log(1e-6 + neg_p.clamp(max = 0.999))) - torch.sum(torch.log(1 + 1e-6 - pos_p.clamp(min = 0.001))))/2
                        loss_shared = loss_shared/images.shape[0]
                    
                    # Shared loss

                    outer_opt.zero_grad()

                    loss = args.lmd_1*loss_cl + args.lmd_2* loss_at + args.lmd_3*loss_shared

                    loss.backward()
                    outer_opt.step()
                    model.eval()

                    # delete the useless variable to save memory
                    del logits, logits_ref, per_logits, per_logits_ref, loss_cl, loss_at, loss_shared, loss

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": round,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })


            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}sau_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}sau_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}sau_df.csv")
        agg.summary().to_csv(f"{args.save_path}sau_df_summary.csv")

        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

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


    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


    def eval_attack(self, netC, net_ref, clean_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in clean_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples
    
    def eval_binary(self, netC, net_ref, bd_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in bd_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    sau.add_arguments(parser)
    args = parser.parse_args()
    sau_method = sau(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = sau_method.defense(args.result_file)