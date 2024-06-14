'''
LIRA: Learnable, Imperceptible and Robust Backdoor Attacks
This is the code for LIRA attack

github link : https://github.com/sunbelbd/invisible_backdoor_attacks

@inproceedings{Doan2021lira,
  title     = {LIRA: Learnable, Imperceptible and Robust Backdoor Attacks},
  author    = {Khoa D. Doan and Yingjie Lao and Weijie Zhao and Ping Li},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year      = {2021}
}

Please note that
1. The original code was implemented in Paddlepaddle,
and we replaced all the functionality of Paddlepaddle with the equivalent API of Pytorch.
2. Since this is a training-controllable attack,
the concepts of poisoning rate and poisoned data may not apply.
So, this attack remains incompatible with the whole framework for the time being
(because we require data to be saved during the saving process).
In the future, we will update this version to make it fully integrated into the framework.
3. For fairness issue, we apply the same total training epochs as all other attack methods. But for LIRA, it may not be the best choice.

The original LICENSE of the script is put at the bottom of this file.
'''
import argparse
import logging
import os
import random
import sys
import torch
from copy import deepcopy
from functools import partial

sys.path = ["./"] + sys.path

import numpy as np
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader

from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization, dataset_and_transform_generate
from utils.trainer_cls import Metric_Aggregator, all_acc, test_given_dataloader_on_mix
from utils.trainer_cls import plot_loss, plot_acc_like_metric, given_dataloader_test
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.save_load_attack import save_attack_result
from attack.badnet import BadNet

loss_fn = nn.CrossEntropyLoss()


class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, (1, 1)),
            nn.BatchNorm2d(out_channel),
        )

        self.out_layer = nn.Tanh()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.concat([x, conv3], 1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.concat([x, conv2], 1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.concat([x, conv1], 1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = self.out_layer(out)

        return out


def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target


def all2all_target_transform(x, num_classes):
    return (x + 1) % num_classes


def create_models(args):
    if args.attack_model == 'autoencoder':
        logging.debug("use autoencoder as atk model")
        atkmodel = Autoencoder(args.input_channel)
        # Copy of attack model
        tgtmodel = Autoencoder(args.input_channel)
    else:
        logging.debug("use unet as atk model")
        atkmodel = UNet(args.input_channel)
        # Copy of attack model
        tgtmodel = UNet(args.input_channel)

    # Classifier
    create_net = partial(generate_cls_model,
                         model_name=args.model,
                         num_classes=args.num_classes,
                         image_size=args.img_size,
                         )
    clsmodel = create_net()

    tgtmodel.to(device)
    atkmodel.to(device)
    clsmodel.to(device)

    # Optimizer
    tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=args.lr_atk)

    return atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net


def hp_test(args, atkmodel, scratchmodel, target_transform,
            train_loader, test_loader, epoch, trainepoch, clip_image,
            testoptimizer=None, log_prefix='Internal', epochs_per_test=5):
    # default phase 2 parameters to phase 1
    if args.test_alpha is None:
        args.test_alpha = args.alpha
    if args.test_eps is None:
        args.test_eps = args.eps

    test_loss = 0
    correct = 0

    correct_transform = 0
    test_transform_loss = 0

    atkmodel.eval()
    if testoptimizer is None:
        scratchmodel.cuda()
        testoptimizer = torch.optim.SGD(params=scratchmodel.parameters(), lr=args.lr)

    for cepoch in range(trainepoch):
        scratchmodel.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            scratchmodel.cuda()
            atkmodel.cuda()
            data, target = data.cuda(), target.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)

                atkoutput = scratchmodel(atkdata)
                output = scratchmodel(data)

                loss_clean = loss_fn(output, target)
                loss_poison = loss_fn(atkoutput, target_transform(target))

                loss = args.alpha * loss_clean + (1 - args.test_alpha) * loss_poison
            scaler.scale(loss).backward()
            scaler.step(testoptimizer)
            scaler.update()
            testoptimizer.zero_grad()

        if cepoch % epochs_per_test == 0 or cepoch == trainepoch - 1:
            scratchmodel.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    scratchmodel.cuda()
                    atkmodel.cuda()
                    data, target = data.cuda(), target.cuda()


                    output = scratchmodel(data)
                    cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
                    test_loss += cross_entropy(output, target).item()  # sum up batch loss
                    correct += (torch.max(output, -1)[1]).eq(target).sum().item()

                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    test_transform_loss += cross_entropy(
                        atkoutput, target_transform(target)).item()  # sum up batch loss
                    correct_transform += (torch.max(atkoutput, -1)[1]).eq(target_transform(target)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)

            logging.debug(
                '\n{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
                    log_prefix, cepoch,
                    test_loss, test_transform_loss,
                    correct, correct_transform
                ))

    return correct, correct_transform


def train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
          train_loader, epoch, train_epoch, create_net, clip_image, post_transforms=None):
    clsmodel.train()
    atkmodel.eval()
    tgtmodel.train()
    losslist = []
    for batch_idx, (data, target) in enumerate(train_loader):

        tgtmodel.cuda()
        clsmodel.cuda()
        atkmodel.cuda()
        data, target = data.cuda(), target.cuda()
        with torch.cuda.amp.autocast(enabled=args.amp):
            if post_transforms is not None:
                data = post_transforms(data)

            ########################################
            #### Update Transformation Function ####
            ########################################
            noise = tgtmodel(data) * args.eps
            atkdata = clip_image(data + noise)

            # Calculate loss
            atkoutput = clsmodel(atkdata)
            loss_poison = loss_fn(atkoutput, target_transform(target))
            loss1 = loss_poison

        losslist.append(loss1.item())

        scaler.scale(loss1).backward()
        scaler.step(tgtoptimizer)
        scaler.update()
        tgtoptimizer.zero_grad()

        ###############################
        #### Update the classifier ####
        ###############################
        with torch.cuda.amp.autocast(enabled=args.amp):
            noise = atkmodel(data) * args.eps
            atkdata = clip_image(data + noise)
            output = clsmodel(data)
            atkoutput = clsmodel(atkdata)
            loss_clean = loss_fn(output, target)
            loss_poison = loss_fn(atkoutput, target_transform(target))
            loss2 = loss_clean * args.alpha + (1 - args.alpha) * loss_poison

        scaler.scale(loss2).backward()
        scaler.step(clsoptimizer)
        scaler.update()
        clsoptimizer.zero_grad()

    atkloss = sum(losslist) / len(losslist)

    return atkloss


def get_target_transform(args):
    """DONE
    """
    if args.attack_label_trans == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args.attack_target)
    elif args.attack_label_trans == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args.num_classes)
    else:
        raise Exception(f'Invalid mode {args.mode}')
    return target_transform


# done
def get_train_test_loaders(args):
    train_loader = get_dataloader(args, True)
    test_loader = get_dataloader(args, False)
    if args.dataset in ['tiny-imagenet', 'tiny-imagenet32', 'tiny']:
        xmin, xmax = -2.1179039478302, 2.640000104904175

        def clip_image(x):
            return torch.clip(x, xmin, xmax)
    elif args.dataset == 'cifar10':
        xmin, xmax = -1.9895, 2.1309

        def clip_image(x):
            return torch.clip(x, xmin, xmax)
    elif args.dataset == 'cifar100':
        xmin, xmax = -1.8974, 2.0243

        def clip_image(x):
            return torch.clip(x, xmin, xmax)
    elif args.dataset == 'mnist':
        def clip_image(x):
            return torch.clip(x, -1.0, 1.0)
    elif args.dataset == 'gtsrb':
        def clip_image(x):
            return torch.clip(x, 0.0, 1.0)
    else:
        raise Exception(f'Invalid dataset: {args.dataset}')
    return train_loader, test_loader, clip_image


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super().__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop)
        self.random_rotation = transforms.RandomRotation(opt.random_rotation)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


# my done
def get_dataloader(opt, train=True, min_width=0):
    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(opt)

    logging.debug("dataset_and_transform_generate done")

    clean_train_dataset_with_transform = dataset_wrapper_with_transform(
        train_dataset_without_transform,
        test_img_transform,  # since later we have post_transform which contains randomcrop etc
        train_label_transform
    )

    clean_test_dataset_with_transform = dataset_wrapper_with_transform(
        test_dataset_without_transform,
        test_img_transform,
        test_label_transform,
    )

    if train:
        dataloader = torch.utils.data.DataLoader(
            clean_train_dataset_with_transform, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            clean_test_dataset_with_transform, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return dataloader


def get_model(args):
    create_net = partial(generate_cls_model,
                         model_name=args.model,
                         num_classes=args.num_classes,
                         image_size=args.img_size,
                         )
    netC = create_net()
    netC.to(device)
    optimizerC, schedulerC = argparser_opt_scheduler(netC, args)
    logging.info(f'atk stage1, Optimizer: {optimizerC}, scheduler: {schedulerC}')
    return netC, optimizerC, schedulerC


def final_test(clean_test_dataloader, bd_test_dataloader, args, test_model_path, atkmodel, netC, target_transform,
               train_loader, test_loader,
               trainepoch, writer, alpha=0.5, optimizerC=None,
               schedulerC=None, log_prefix='Internal', epochs_per_test=1, data_transforms=None, start_epoch=1,
               clip_image=None, criterion=None):
    atkmodel.cuda()
    netC.cuda()

    clean_accs, poison_accs = [], []

    atkmodel.eval()

    if optimizerC is None:
        logging.debug('No optimizer, creating default SGD...')
        optimizerC = optim.SGD(netC.parameters(), lr=args.finetune_lr)
    if schedulerC is None:
        logging.debug('No scheduler, creating default 100,200,300,400...')
        schedulerC = optim.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], args.finetune_lr)

    clean_test_loss_list = []
    bd_test_loss_list = []
    test_acc_list = []
    test_asr_list = []
    test_ra_list = []
    train_loss_list = []

    agg = Metric_Aggregator()

    for cepoch in range(start_epoch, trainepoch + 1):
        netC.train()
        batch_loss_list = []
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()
            netC.cuda()
            atkmodel.cuda()

            with torch.cuda.amp.autocast(enabled=args.amp):

                if data_transforms is not None:
                    data = data_transforms(data)

                output = netC(data)
                loss_clean = loss_fn(output, target)

                if alpha < 1:
                    with torch.no_grad():
                        noise = atkmodel(data) * args.test_eps
                        if clip_image is None:
                            atkdata = torch.clip(data + noise, 0, 1)
                        else:
                            atkdata = clip_image(data + noise)
                    atkoutput = netC(atkdata)
                    loss_poison = loss_fn(atkoutput, target_transform(target))
                else:
                    loss_poison = torch.tensor(0.0)

                loss = alpha * loss_clean + (1 - alpha) * loss_poison

            scaler.scale(loss).backward()
            scaler.step(optimizerC)
            scaler.update()
            optimizerC.zero_grad()

            batch_loss_list.append(loss.item())
        one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        schedulerC.step()

        # test

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
                "epoch": cepoch,
                "train_epoch_loss_avg_over_batch": one_epoch_loss,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            }
        )

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
            "loss_metric_plots_atk_stg2",
        )

        plot_acc_like_metric(
            [], [], [],
            test_acc_list,
            test_asr_list,
            test_ra_list,
            args.save_path,
            "loss_metric_plots_atk_stg2",
        )

        agg.to_dataframe().to_csv(f"{args.save_path}/attack_df_atk_stg2.csv")

    agg.summary().to_csv(f"{args.save_path}/attack_df_summary_atk_stg2.csv")

    return clean_accs, poison_accs


def main(args, clean_test_dataset_with_transform, criterion):
    clean_test_dataloader = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                       drop_last=False,
                                       pin_memory=args.pin_memory, num_workers=args.num_workers, )

    clean_test_loss_list = []
    bd_test_loss_list = []
    test_acc_list = []
    test_asr_list = []
    test_ra_list = []
    train_loss_list = []

    agg = Metric_Aggregator()

    if args.verbose >= 1:
        logging.debug('========== ARGS ==========')
        logging.debug(args)

    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args)

    logging.debug('========== DATA ==========')
    logging.debug('Loaders: Train {} examples/{} iters, Test {} examples/{} iters'.format(
        len(train_loader.dataset), len(train_loader), len(test_loader.dataset), len(test_loader)))

    atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net = create_models(args)
    if args.verbose >= 2:
        logging.debug('========== MODELS ==========')
        logging.debug(atkmodel)
        logging.debug(clsmodel)

    target_transform = get_target_transform(args)

    trainlosses = []
    start_epoch = 1

    # Initialize the tgtmodel
    tgtmodel.load_state_dict(atkmodel.state_dict(), )

    logging.debug('============================')
    logging.debug('============================')

    logging.debug('BEGIN TRAINING >>>>>>')
    clsmodel.cuda()
    clsoptimizer = torch.optim.SGD(params=clsmodel.parameters(), lr=args.lr, momentum=0.9)
    if "," in args.device:
        logging.info("data parallel in main")
        atkmodel = torch.nn.DataParallel(
            atkmodel,
            device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        )
        tgtmodel = torch.nn.DataParallel(
            tgtmodel,
            device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        )
        clsmodel = torch.nn.DataParallel(
            clsmodel,
            device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        )

    for epoch in range(start_epoch, args.both_train_epochs + 1):
        for i in range(args.train_epoch):
            logging.debug(f'===== EPOCH: {epoch}/{args.both_train_epochs + 1} CLS {i + 1}/{args.train_epoch} =====')
            if not args.avoid_clsmodel_reinitialization:
                clsmodel.cuda()
                clsoptimizer = torch.optim.SGD(params=clsmodel.parameters(), lr=args.lr)
            trainloss = train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
                              train_loader,
                              epoch, i, create_net, clip_image,
                              post_transforms=post_transforms)
            trainlosses.append(trainloss)
        atkmodel.load_state_dict(tgtmodel.state_dict())
        if args.avoid_clsmodel_reinitialization:
            scratchmodel = create_net()
            if "," in args.device:
                scratchmodel = torch.nn.DataParallel(
                    scratchmodel,
                    device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
                )
            scratchmodel.load_state_dict(clsmodel.state_dict())  # transfer from cls to scratch for testing
        else:
            clsmodel = create_net()
        # test

        clsmodel.eval()
        clsmodel.to(device, non_blocking=args.non_blocking)
        atkmodel.eval()
        atkmodel.to(device, non_blocking=args.non_blocking)

        bd_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform.wrapped_dataset.dataset,
            save_folder_path=f"{args.save_path}/bd_test_dataset_stage_one"
        )
        for trans_t in deepcopy(
                clean_test_dataset_with_transform.wrap_img_transform.transforms):
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
                logging.info(f"{denormalizer}")

        clean_test_dataset_with_transform.wrapped_dataset.getitem_all = True

        for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(
                clean_test_dataloader):
            with torch.no_grad():
                x = x.to(device, non_blocking=args.non_blocking)

                noise = atkmodel(x) * args.test_eps
                atkdata = clip_image(x + noise)
                atktarget = target_transform(labels)

                position_changed = (labels - atktarget != 0)
                atkdata = atkdata[position_changed]
                targets1 = labels.detach().clone().cpu()[position_changed]
                y_poison_batch = atktarget.detach().clone().cpu()[position_changed]
                for idx_in_batch, t_img in enumerate(atkdata.detach().clone().cpu()):
                    if int(y_poison_batch[idx_in_batch]) == int(targets1[idx_in_batch]):
                        print("find bug")
                    bd_test_dataset.set_one_bd_sample(
                        selected_index=int(
                            batch_idx * int(args.batch_size) + torch.where(position_changed.detach().clone().cpu())[0][
                                idx_in_batch]),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=denormalizer(t_img),
                        bd_label=int(y_poison_batch[idx_in_batch]),
                        label=int(targets1[idx_in_batch]),
                    )

        bd_test_dataset.subset(
            np.where(bd_test_dataset.poison_indicator == 1)[0]
        )
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            clean_test_dataset_with_transform.wrap_img_transform,
        )

        # generate bd test dataloader

        bd_test_dataloader = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False,
                                        pin_memory=args.pin_memory, num_workers=args.num_workers, )

        ### My test code start

        clean_metrics, \
        clean_test_epoch_predict_list, \
        clean_test_epoch_label_list, \
            = given_dataloader_test(
            model=clsmodel,
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
            model=clsmodel,
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
                "train_epoch_loss_avg_over_batch": trainloss,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            }
        )

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
            "loss_metric_plots_atk_stg1",
        )

        plot_acc_like_metric(
            [], [], [],
            test_acc_list,
            test_asr_list,
            test_ra_list,
            args.save_path,
            "loss_metric_plots_atk_stg1",
        )

        agg.to_dataframe().to_csv(f"{args.save_path}/attack_df_atk_stg1.csv")

    agg.summary().to_csv(f"{args.save_path}/attack_df_summary_atk_stg1.csv")

    ### My test code end

    return clsmodel, atkmodel, bd_test_dataloader


def main2(args, pre_clsmodel, pre_atkmodel, clean_test_dataloader, bd_test_dataloader, criterion):
    args_for_finetune = deepcopy(args)
    args_for_finetune.__dict__ = {k[9:]: v for k, v in args_for_finetune.__dict__.items() if k.startswith("finetune_")}

    if args.test_alpha is None:
        logging.debug(f'Defaulting test_alpha to train alpha of {args.alpha}')
        args.test_alpha = args.alpha

    if args.finetune_lr is None:
        logging.debug(f'Defaulting test_lr to train lr {args.lr}')
        args.finetune_lr = args.lr

    if args.test_eps is None:
        logging.debug(f'Defaulting test_eps to train eps {args.test_eps}')
        args.test_eps = args.eps

    logging.debug('====> ARGS')
    logging.debug(args)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, clip_image = get_train_test_loaders(args)

    atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)
    netC, optimizerC, schedulerC = get_model(args)

    if "," in args.device:
        logging.info("data parallel in main2")
        netC = torch.nn.DataParallel(
            netC,
            device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        )
        atkmodel = torch.nn.DataParallel(
            atkmodel,
            device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        )

    netC.load_state_dict(pre_clsmodel.state_dict())

    target_transform = get_target_transform(args)

    atkmodel.load_state_dict(pre_atkmodel.state_dict())

    netC.to(device)

    optimizerC, schedulerC = argparser_opt_scheduler(netC, args_for_finetune)

    logging.debug(netC)
    logging.info(f"atk stage2, optimizerC: {optimizerC}, schedulerC: {schedulerC}")

    data_transforms = PostTensorTransform(args)
    logging.debug('====> Post tensor transform')
    logging.debug(data_transforms)

    clean_accs, poison_accs = final_test(clean_test_dataloader, bd_test_dataloader,
                                         args, None, atkmodel, netC, target_transform,
                                         train_loader, test_loader,
                                         trainepoch=int(args.epochs - args.both_train_epochs),
                                         writer=None, log_prefix='POISON', alpha=args.test_alpha, epochs_per_test=1,
                                         optimizerC=optimizerC, schedulerC=schedulerC, data_transforms=data_transforms,
                                         clip_image=clip_image, criterion=criterion)


class LIRA(BadNet):

    def __init__(self):
        super(LIRA, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--attack', type=str, )
        parser.add_argument('--attack_target', type=int,
                            help='target class in all2one attack')
        parser.add_argument('--attack_label_trans', type=str,
                            help='which type of label modification in backdoor attack'
                            )
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/lira/default.yaml',
                            help='path for yaml file provide additional default attributes')

        parser.add_argument("--random_crop", type=int, )
        parser.add_argument("--random_rotation", type=int, )

        parser.add_argument('--attack_model', type=str, )  # default='autoencoder')
        parser.add_argument('--lr_atk', type=float, )  # default=0.0001, help='learning rate for attack model')
        parser.add_argument('--eps', type=float, )  # default=0.3, help='epsilon for data poisoning')
        parser.add_argument('--test_eps', type=float)  # default=None,
        parser.add_argument('--alpha', type=float, )  # default=0.5)
        parser.add_argument('--test_alpha', type=float)  # default=None,
        parser.add_argument('--fix_generator_epoch', type=int, )  # default=1, help='training epochs for victim model')

        # parser.add_argument('--finetune_epochs', type=int)  # default=500,
        parser.add_argument('--finetune_lr', type=float)  # default=None,

        # parser.add_argument('--steplr_gamma', )  # default='30,60,90,150')
        # parser.add_argument('--steplr_milestones', type=float)  # default=0.05,
        parser.add_argument('--finetune_steplr_gamma', )
        parser.add_argument('--finetune_steplr_milestones', )
        parser.add_argument('--finetune_optimizer', )  # default='sgd')
        parser.add_argument("--both_train_epochs", type=int, )

        ################### its original
        parser.add_argument('--train_epoch', type=int, default=1, help='training epochs for victim model')
        parser.add_argument('--epochs_per_external_eval', type=int, default=50)
        parser.add_argument('--best_threshold', type=float, default=0.1)
        parser.add_argument('--verbose', type=int, default=1, help='verbosity')
        parser.add_argument('--avoid_clsmodel_reinitialization', action='store_true',
                            default=True, help='whether test the poisoned model from scratch')
        parser.add_argument('--test_n_size', default=10)
        parser.add_argument('--test_use_train_best', default=False, action='store_true')
        parser.add_argument('--test_use_train_last', default=True, action='store_true')

        return parser

    def stage1_non_training_data_prepare(self):
        pass

    def stage2_training(self):
        logging.info(f"stage2 start")
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

        clean_test_dataset = prepro_cls_DatasetBD_v2(
            test_dataset_without_transform
        )
        clean_test_dataset_with_transform = dataset_wrapper_with_transform(clean_test_dataset, test_img_transform)
        clean_test_dataloader = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                           drop_last=False,
                                           pin_memory=args.pin_memory, num_workers=args.num_workers, )

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


        criterion = argparser_criterion(args)

        global device
        device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )
        global scaler
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        clsmodel, atkmodel, bd_test_dataloader = main(args, clean_test_dataset_with_transform, criterion)
        main2(args, clsmodel, atkmodel, clean_test_dataloader, bd_test_dataloader, criterion)
        ###

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=clsmodel.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=None,
            bd_test=bd_test_dataloader.dataset.wrapped_dataset,
            save_path=args.save_path,
        )


if __name__ == '__main__':
    attack = LIRA()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    assert int(args.epochs - args.both_train_epochs) > 0, "(total) epochs should be larger than both_train_epochs"
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

'''
MIT License

Copyright (c) 2021 Cognitive Computing Lab

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
