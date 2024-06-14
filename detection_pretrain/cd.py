'''
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import csv
import os,sys
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
# from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization, get_dataset_normalization, get_input_shape, get_num_classes, get_transform

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from cvpr_cd.detection import CognitiveDistillation

def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


class CognitiveDistillation(nn.Module):
    def __init__(self, args, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.args = args
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        transforms_list = []
        dataset_normalization = get_dataset_normalization(self.args.dataset)
        transforms_list.append(dataset_normalization)
        tran = transforms.Compose(transforms_list)
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            features, logits = model(images)
        else:
            logits = model(images).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * tran(torch.rand(b, c, 1, 1)).to(images.device)
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()

class ft:
    r"""Basic class for ft defense method.
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor model
    4. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        ft.add_arguments(parser)
        args = parser.parse_args()
        ft_method = ft(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = ft_method.defense(args.result_file)
    
    .. Note::
        

    Args:
        baisc args: in the base class
        ratio (float): the ratio of clean data loader
        index (str): index of clean data
    
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

    def cal(self, true, pred):
        TN, FP, FN, TP = metrics.confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/cd/config.yaml", help='the path of yaml')

        #set the parameter for the ft defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        parser.add_argument('--cd_lr', type=float, help='cd lr')
        parser.add_argument('--cd_p', type=int, help='norm')
        parser.add_argument('--cd_gamma', type=float, help='gamma')
        parser.add_argument('--cd_beta', type=float, help='beta')
        parser.add_argument('--cd_num_steps', type=int, help='num_steps')
        parser.add_argument('--csv_save_path', type=str, help='the path of csv')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection_pretrain/cd/'
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
    def filtering(self):
        start = time.perf_counter()
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
       
        model.eval()
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(args)

        

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        clean_dataset = self.result['bd_train'].wrapped_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = clean_dataset
        data_set_o.wrap_img_transform = test_tran
        # data_set_o = prepro_cls_DatasetBD_v2(
        #     full_dataset_without_transform=data_set,
        #     poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        #     bd_image_pre_transform=None,
        #     bd_label_pre_transform=None,
        #     ori_image_transform_in_loading=train_tran,
        #     ori_label_transform_in_loading=None,
        #     add_details_in_preprocess=False,
        # )
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False, pin_memory=args.pin_memory)
        trainloader = data_loader
        pindex = np.where(np.array(clean_dataset.poison_indicator) == 1)[0]
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False,pin_memory=args.pin_memory)

        # self.trainer.train_with_test_each_epoch(
        #     train_data = trainloader,
        #     test_data = data_clean_loader,
        #     adv_test_data = data_bd_loader,
        #     end_epoch_num = self.args.epochs,
        #     criterion = criterion,
        #     optimizer = optimizer,
        #     scheduler = scheduler,
        #     device = self.args.device,
        #     frequency_save = self.args.frequency_save,
        #     save_folder_path = self.args.checkpoint_save,
        #     save_prefix = 'defense',
        #     continue_training_path = None,
        # )

        cd = CognitiveDistillation(args, lr=args.cd_lr, p=args.cd_p, gamma=args.cd_gamma, beta=args.cd_beta, num_steps=args.cd_num_steps, norm_only=True)

        mask_train = []
        mask_test_clean = []
        mask_test_bd = []

        train_poison = []

        train_label = []
        test_clean_label = []
        test_bd_label = []

        for batch_idx, (inputs, targets, *add) in enumerate(trainloader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            mask = cd(model, inputs)
            mask = mask.detach().cpu().numpy().reshape(-1)
            mask = [i for i in mask]
            mask_train.extend(mask)
            add_infor = add[1].numpy().reshape(-1)
            add_infor = [i for i in add_infor]
            train_poison.extend(add_infor)
            targets_label = targets.detach().cpu().numpy().reshape(-1)
            targets_label = [i for i in targets_label]
            train_label.extend(targets_label)

        for batch_idx, (inputs, targets) in enumerate(data_clean_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            mask = cd(model, inputs)
            mask = mask.detach().cpu().numpy().reshape(-1)
            mask = [i for i in mask]
            mask_test_clean.extend(mask)
            targets_label = targets.detach().cpu().numpy().reshape(-1)
            targets_label = [i for i in targets_label]
            test_clean_label.extend(targets_label)

        for batch_idx, (inputs, targets, *add) in enumerate(data_bd_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            mask = cd(model, inputs)
            mask = mask.detach().cpu().numpy().reshape(-1)
            mask = [i for i in mask]
            mask_test_bd.extend(mask)
            targets_label = targets.detach().cpu().numpy().reshape(-1)
            targets_label = [i for i in targets_label]
            test_bd_label.extend(targets.detach().cpu().numpy()) 
        torch.save({
            'mask_train': mask_train,
            'mask_test_clean': mask_test_clean,
            'mask_test_bd': mask_test_bd,
            'train_poison': train_poison,
            'train_label': train_label,
            'test_clean_label': test_clean_label,
            'test_bd_label': test_bd_label
        }, f'{self.args.checkpoint_save}cd_results.pt')

        ### calculate the TPR and FPR and AUC
        ### calculate the threshold
        mask_train_array = np.array(mask_train)
        mask_clean_train = np.array(mask_train)[np.array(train_poison) == 0]
        ### random choose the clean data
        ratio_clean = 0.05
        mask_clean_train = np.random.choice(mask_clean_train, int(len(mask_clean_train)*ratio_clean), replace=False)
        ### calculate the mean and std
        mean = np.mean(mask_clean_train)
        std = np.std(mask_clean_train)
        ### calculate the threshold
        threshold = mean - std
        suspect_index = np.where(mask_train_array > threshold)[0]

        true_index = np.zeros(len(mask_train))
        label_pred = []
        mask_train_max = np.max(mask_train)
        mask_train_min = np.min(mask_train)
        for i in range(len(mask_train)):
            label_pred_i = (mask_train[i] - mask_train_min)/(mask_train_max - mask_train_min)
            label_pred_i = 1 - label_pred_i
            label_pred.append(label_pred_i)

        fpr, tpr, thresholds = metrics.roc_curve(train_poison, label_pred, pos_label=1)

        f = open(self.args.save_path + '/fpr_tpr.txt', 'w', encoding='utf-8')
        for i,j in zip(fpr, tpr):
            f.write(str(i) + ',' + str(j) + '\n')
        f.close()
        
        for i,j in zip(fpr, tpr):
            if i <= 0.20:
                TPR = j
                FPR = i
        auc = metrics.auc(fpr, tpr)

        for i in range(len(true_index)):
            if i in pindex:
                true_index[i] = 1
        if len(suspect_index)==0:
            tn = len(true_index) - np.sum(true_index)
            fp = np.sum(true_index)
            fn = 0
            tp = 0
            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'AUC', 'target'])
            csv_write.writerow([args.result_file, tn,fp,fn,tp, 0,0, auc, 'None'])
            f.close()
            f = open(args.csv_save_path, 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow([args.result_file, 'cd', 'None', 'None', tn,fp,fn,tp, 0,0, 0, auc, 'None'])
            f.close()
        else: 
            findex = np.zeros(len(mask_train))
            for i in range(len(findex)):
                if i in suspect_index:
                    findex[i] = 1
            tn, fp, fn, tp = self.cal(true_index, findex)
            # TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)

            new_TP = tp
            new_FN = fn*9
            new_FP = fp*1
            precision = new_TP / (new_TP + new_FP) if new_TP + new_FP != 0 else 0
            recall = new_TP / (new_TP + new_FN) if new_TP + new_FN != 0 else 0
            fw1 = 2*(precision * recall)/ (precision + recall) if precision + recall != 0 else 0
            end = time.perf_counter()
            time_miniute = (end-start)/60

            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'AUC', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, auc, 'None'])
            f.close()

            f = open(args.csv_save_path, 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow([args.result_file, 'cd', 'None', 'None', tn, fp, fn, tp, TPR, FPR, fw1, auc, time_miniute, 'None'])
            f.close()



    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        self.filtering()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ft.add_arguments(parser)
    args = parser.parse_args()
    ft_method = ft(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ft_method.detection(args.result_file)