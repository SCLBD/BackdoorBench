'''
Poison ink: Robust and invisible backdoor attack
This script is for Poison ink attack.

code : https://github.com/ZJZAC/Poison-Ink

@article{zhang2022poison,
  title={Poison ink: Robust and invisible backdoor attack},
  author={Zhang, Jie and Dongdong, Chen and Huang, Qidong and Liao, Jing and Zhang, Weiming and Feng, Huamin and Hua, Gang and Yu, Nenghai},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={5691--5705},
  year={2022},
  publisher={IEEE}
}

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense

Note that the autoencoder training process and img process part are not in this script,
    which are time comsume and dataset-dependent, please follow https://github.com/ZJZAC/Poison-Ink to train models for generating the poisoned data.
    (We also provide the training and generation script, whose protocols are rewriten for our benchmark in resource/poison-ink, please go there and check the readme file)
    Then please use the model to generate poisoned version for all train and test data, then save to args.attack_train_replace_imgs_path
    and args.attack_test_replace_imgs_path (path you use in args of this script). (Format is "train/{class_idx}/img_{img_idx}.png" and "test/{class_idx}/img_{img_idx}.png")
    After that, you can use this script to train the poisoned model.

Also notice that this is a training-controllable attack.

LICENSE is at the end of this file.
'''

import argparse
import logging
import os
import sys

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape


import os
import sys
import yaml

sys.path = ["./"] + sys.path

import argparse
import random
import numpy as np
import torch
import logging
from math import sqrt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.aggregate_block.bd_attack_generate import general_compose
from utils.bd_dataset_v2 import x_iter
from utils.trainer_cls import given_dataloader_test, Metric_Aggregator
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization, dataset_and_transform_generate
from utils.trainer_cls import Metric_Aggregator, all_acc, test_given_dataloader_on_mix
from utils.trainer_cls import plot_loss, plot_acc_like_metric, given_dataloader_test
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.save_load_attack import save_attack_result
from attack.badnet import BadNet
from time import time

def get_files(directory, exts=(".png",)):
    ext_extensions = exts  # Add any other image extensions you want to include
    ext_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(ext_extensions):
                file_abs_path = os.path.abspath(os.path.join(root, file))
                # print(f"load file from {file_abs_path}")
                ext_files.append(
                    # Image.open(file_abs_path)
                    file_abs_path
                )
    return ext_files

class attack_replace_version(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, replace_image_paths: dict) -> None:

        self.replace_image_paths = replace_image_paths

    def __call__(self, img: None,
                 target: None,
                 image_serial_id: int
                 ) :
        img = (Image.open(self.replace_image_paths[image_serial_id]))
        img_r =  deepcopy(img)
        img.close()
        return img_r

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def original_train_one_epoch(trainloader, wminputs, wmtargets , model, criterion, optimizer, epoch, device=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time()

    # bar = Bar('Processing', max=len(trainloader))

    wm_id = np.random.randint(len(wminputs))
    for batch_idx, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time() - end)
        wm_input = wminputs[(wm_id + batch_idx) % len(wminputs)]
        wm_target = wmtargets[(wm_id + batch_idx) % len(wmtargets)]

        if device:
            input, target = input.to(device), target.to(device)
            wm_input, wm_target = wm_input.to(device), wm_target.to(device)

        inputs = torch.cat([input, wm_input], dim=0)
        targets = torch.cat([target, wm_target], dim=0)

        # print(targets.shape)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

    #     # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
    #                   '| Loss_combine: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(trainloader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #                 )
    #     bar.next()
    # bar.finish()
    return losses.avg, top1.avg

class poison_ink(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

        parser = add_common_attack_args(parser)
        parser.add_argument('--attack_train_replace_imgs_path', type=str)
        parser.add_argument('--attack_test_replace_imgs_path', type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/poison_ink/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def process_args(self, args):

        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        if ('attack_train_replace_imgs_path' not in args.__dict__) or (args.attack_train_replace_imgs_path is None):
            args.attack_train_replace_imgs_path = f"./resource/poison_ink/train"
            logging.info(
                f"args.attack_train_replace_imgs_path does not found, so = {args.attack_train_replace_imgs_path}")

        if ('attack_test_replace_imgs_path' not in args.__dict__) or (args.attack_test_replace_imgs_path is None):
            args.attack_test_replace_imgs_path = f"./resource/poison_ink/test"
            logging.info(
                f"args.attack_test_replace_imgs_path does not found, so = {args.attack_test_replace_imgs_path}")

        return args

    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        train_bd_replace_images = get_files(args.attack_train_replace_imgs_path)
        train_bd_idx_imgs_map = {}
        for img_path in train_bd_replace_images:
            img_name = img_path.split('/')[-1]
            img_idx = img_name.split('.')[0].replace("img_","")
            train_bd_idx_imgs_map[int(img_idx)] = img_path

        logging.info(f"train_bd_replace_images {len(train_bd_replace_images)}")

        test_bd_replace_images = get_files(args.attack_test_replace_imgs_path)
        test_bd_idx_imgs_map = {}
        for img_path in test_bd_replace_images:
            img_name = img_path.split('/')[-1]
            img_idx = img_name.split('.')[0].replace("img_","")
            test_bd_idx_imgs_map[int(img_idx)] = img_path

        logging.info(f"test_bd_replace_images {len(test_bd_replace_images)}")

        train_bd_img_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (attack_replace_version(
                train_bd_idx_imgs_map
            ), True),
        ])
        test_bd_img_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (attack_replace_version(
                test_bd_idx_imgs_map
            ), True),
        ])
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)

        ### 4. set the backdoor attack data and backdoor test data
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
        )

        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle',
                   )

        ### generate train dataset for backdoor attack
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            test_img_transform, # important, the source code comment out all other transformations, use the same tranfromation as test phase.
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = \
              clean_train_dataset_with_transform, \
              clean_test_dataset_with_transform, \
              bd_train_dataset_with_transform, \
              bd_test_dataset_with_transform

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        trainloader = DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # testloader = DataLoader(val_dataset,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        bd_train_dataset_with_transform.wrapped_dataset.getitem_all = False
        wm_trainloader = DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        #     wm_testloader = DataLoader(val_dataset_wm,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        wminputs, wmtargets = [], []
        for wm_idx, (wminput, wmtarget) in enumerate(wm_trainloader):
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # Model
        # print("==> creating model '{}'".format(args.arch))
        # if args.arch.startswith('resnext'):
        #     model = models.__dict__[args.arch](
        #                 cardinality=args.cardinality,
        #                 num_classes=num_classes,
        #                 depth=args.depth,
        #                 widen_factor=args.widen_factor,
        #                 dropRate=args.drop,
        #             )
        # elif args.arch.startswith('densenet'):
        #     model = models.__dict__[args.arch](
        #                 num_classes=num_classes,
        #                 depth=args.depth,
        #                 growthRate=args.growthRate,
        #                 compressionRate=args.compressionRate,
        #                 dropRate=args.drop,
        #             )
        # elif args.arch.startswith('wrn'):
        #     model = models.__dict__[args.arch](
        #                 num_classes=num_classes,
        #                 depth=args.depth,
        #                 widen_factor=args.widen_factor,
        #                 dropRate=args.drop,
        #             )
        # elif args.arch.endswith('resnet'):
        #     model = models.__dict__[args.arch](
        #                 num_classes=num_classes,
        #                 depth=args.depth,
        #                 block_name=args.block_name,
        #             )
        # else:
        #     model = models.__dict__[args.arch](num_classes=num_classes)

        # model = torch.nn.DataParallel(model).cuda()
        model = self.net.cuda()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # # Resume
        # if args.dataset == "cifar10":
        #     title = 'cifar-10-' + '-T-' + args.task #'cifar-10-' + args.arch + '-T-' + args.task
        # elif args.dataset == "cifar100":
        #     title = 'cifar-100-' + '-T-' + args.task #'cifar-100-' + args.arch + '-T-' + args.task

        # if False:
        #     pass
        #     # # Load checkpoint.
        #     # print('==> Resuming from checkpoint..')
        #     # assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        #     # args.checkpoint = os.path.dirname(args.resume)
        #     # checkpoint = torch.load(args.resume)
        #     # best_acc = checkpoint['best_acc']
        #     # start_epoch = checkpoint['epoch']
        #     # model.load_state_dict(checkpoint['state_dict'])
        #     # optimizer.load_state_dict(checkpoint['optimizer'])
        #     # logger_loss = Logger(os.path.join(args.checkpoint, 'log_loss.txt'), title=title, resume=True)
        #     # logger_acc = Logger(os.path.join(args.checkpoint, 'log_acc.txt'), title=title, resume=True)
        #
        # else:
        #     logger_loss = Logger(os.path.join(args.checkpoint, 'loss.txt'), title=title)
        #     logger_loss.set_names([ 'Train Combine Loss', 'Valid Clean Loss',  'Valid Trigger Loss.'])
        #     logger_acc = Logger(os.path.join(args.checkpoint, 'acc.txt'), title=title)
        #     logger_acc.set_names([ ' Train Combine Acc ', ' Valid Clean  Acc.',  ' Valid Trigger Acc.'])

        # if args.evaluate:
        #     print('\nEvaluation only')
        #     test_loss1, test_acc, test_loss2, test_wm = test(testloader, model, criterion, start_epoch, use_cuda)
        #     print(' Test Loss:  %.8f, Test Acc:  %.2f, Test Wm Acc:  %.2f' % (test_loss, test_acc, test_wm))
        #     return

        device = self.device
        netC = model
        clean_test_dataloader = DataLoader(
            clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
               drop_last=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers, )
        bd_test_dataloader = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, )

        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        train_loss_list = []
        train_acc_list = []

        agg = Metric_Aggregator()

        # Train and val
        for epoch in range(0, args.epochs):

            train_loss_combine, train_acc_combine = original_train_one_epoch(trainloader, wminputs, wmtargets, model, criterion,
                                                           optimizer, epoch, self.device)
            train_loss_combine, train_acc_combine = train_loss_combine.item(), train_acc_combine.item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # here since ReduceLROnPlateau need the train loss to decide next step setting.
                    scheduler.step(train_loss_combine)
                else:
                    scheduler.step()

            clean_metrics, \
                clean_test_epoch_predict_list, \
                clean_test_epoch_label_list, \
                = given_dataloader_test(
                model=netC,
                test_dataloader=clean_test_dataloader,
                criterion=criterion,
                non_blocking=args.non_blocking,
                device=device,
                verbose=1,
            )

            clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
            test_acc = clean_metrics["test_acc"]

            bd_metrics, \
                bd_test_epoch_predict_list, \
                bd_test_epoch_label_list, \
                bd_test_epoch_original_index_list, \
                bd_test_epoch_poison_indicator_list, \
                bd_test_epoch_original_targets_list = test_given_dataloader_on_mix(
                model=netC,
                test_dataloader=bd_test_dataloader,
                criterion=criterion,
                non_blocking=args.non_blocking,
                device=device,
                verbose=1,
            )

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            agg(
                {
                    "epoch": epoch,
                    "train_epoch_loss_avg_over_batch": train_loss_combine,
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                }
            )

            train_acc_list.append(train_acc_combine)
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            plot_loss(
                train_loss_list,
                clean_test_loss_list,
                bd_test_loss_list,
                args.save_path,
            )

            plot_acc_like_metric(
                train_acc_list, [], [],
                test_acc_list,
                test_asr_list,
                test_ra_list,
                args.save_path,
            )

            agg.to_dataframe().to_csv(f"{args.save_path}/attack_df.csv")

        agg.summary().to_csv(f"{args.save_path}/attack_df_summary.csv")
        # trainer.train_with_test_each_epoch_on_mix(
        #     DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
        #                pin_memory=args.pin_memory, num_workers=args.num_workers, ),
        #     DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
        #                drop_last=False,
        #                pin_memory=args.pin_memory, num_workers=args.num_workers, ),
        #     DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
        #                pin_memory=args.pin_memory, num_workers=args.num_workers, ),
        #     args.epochs,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     device=self.device,
        #     frequency_save=args.frequency_save,
        #     save_folder_path=args.save_path,
        #     save_prefix='attack',
        #     amp=args.amp,
        #     prefetch=args.prefetch,
        #     prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
        #     non_blocking=args.non_blocking,
        # )
        bd_train_dataset_with_transform.wrapped_dataset.getitem_all = True
        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )

if __name__ == '__main__':
    attack = poison_ink()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

'''
MIT License

Copyright (c) 2022 Jie Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''