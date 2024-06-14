'''

Beating Backdoor Attack at Its Own Game

This file is modified based on the following source:
link : https://github.com/SCLBD/DBD & https://github.com/damianliumin/non-adversarial_backdoor

@inproceedings{liu2023beating,
    title={Beating Backdoor Attack at Its Own Game},
    author={Liu, Min and Sangiovanni-Vincentelli, Alberto and Yue, Xiangyu},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={4620--4629},
    year={2023}}

The defense method is called nab.
The license is bellow the code

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. add some new backdone such as mobilenet efficientnet and densenet, reconstruct the backbone of vgg and preactresnet
    7. Different data augmentation (transform) methods are used
    8. rewrite the dateset
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. nab defense:
        a. self-supervised learning generates feature extractor
        b. LGA from ABL method to detect poison samples
        c. relabel the detected samples
        d. train the model using the relabelled dataset
    4. test the result and get ASR, ACC, RA 
    
Note:
    The original code use an additional clean dataset to train a auxiliary classifier for relabeling. 
    To make a fair comparison, we use the SSL model from DBD for relabeling as described in the paper.
'''


import logging
import time
import argparse
import sys
import os


sys.path.append('../')
sys.path.append(os.getcwd())
from utils.defense_utils.dbd.data.prefetch import PrefetchLoader

import numpy as np
import torch
import yaml
from utils.trainer_cls import Metric_Aggregator
from pprint import pformat
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization, get_dataset_denormalization
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch, given_dataloader_test

from utils.defense_utils.dbd.data.dataset import SelfPoisonDataset
from utils.aggregate_block.fix_random import fix_random
from utils.save_load_attack import load_attack_result

from utils.defense_utils.dbd.model.model import SelfModel
from utils.defense_utils.dbd.model.utils import (
    get_network_dbd,
    load_state,
    get_criterion,
    get_optimizer,
    get_scheduler,
)
from utils.bd_dataset_v2 import xy_iter, slice_iter
from utils.defense_utils.dbd.utils_db.setup import (
    load_config,
)
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler

from utils.defense_utils.dbd.utils_db.trainer.simclr import simclr_train
from utils.aggregate_block.dataset_and_transform_generate import get_transform_self

def get_information(args,result,config_ori):
    config = config_ori
    aug_transform = get_transform_self(args.dataset, *([args.input_height,args.input_width]) , train = True, prefetch =args.prefetch)
    
    x = slice_iter(result["bd_train"], axis=0)
    y = slice_iter(result["bd_train"], axis=1)

    self_poison_train_data = SelfPoisonDataset(x,y, aug_transform,args)
    
    self_poison_train_loader_ori = torch.utils.data.DataLoader(self_poison_train_data, batch_size=args.batch_size_self, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    if args.prefetch:
        # x,y: PIL.Image.Image -> SelfPoisonDataset: Tensor with trans, no normalization [0,255]-> PrefetchLoader: Tensor with trans [0,1], with normalization
        self_poison_train_loader = PrefetchLoader(self_poison_train_loader_ori, self_poison_train_data.mean, self_poison_train_data.std)
    else:
        # x,y: PIL.Image.Image, [0,255] -> SelfPoisonDataset: Tensor with trans [0,1] with normalization
        self_poison_train_loader = self_poison_train_loader_ori
        
    backbone = get_network_dbd(args)
    self_model = SelfModel(backbone)
    self_model = self_model.to(args.device)
    criterion = get_criterion(config["criterion"])
    criterion = criterion.to(args.device)
    optimizer = get_optimizer(self_model, config["optimizer"])
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    resumed_epoch = load_state(
        self_model, args.resume, args.checkpoint_load, 0, optimizer, scheduler,
    )
    box = {
      'self_poison_train_loader': self_poison_train_loader,
      'self_model': self_model,
      'criterion': criterion,
      'optimizer': optimizer,
      'scheduler': scheduler,
      'resumed_epoch': resumed_epoch
    }
    return box



def get_args():
    # set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument("--num_workers_semi", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')
    parser.add_argument('--yaml_path', type=str, default="./config/defense/nab/config.yaml", help='the path of yaml')

    # set the parameter for the general
    parser.add_argument('--prefetch',type=bool )

    # SSL part for relabel
    parser.add_argument('--epoch_warmup',type=int )
    parser.add_argument('--batch_size_self',type=int )
    parser.add_argument('--temperature',type=int )
    parser.add_argument('--epsilon',type=int )
    parser.add_argument('--epoch_self',type=int )


    # LGA part for detection
    parser.add_argument('--epoch_lga',  default= 20,type=int )
    parser.add_argument('--gamma',  default= 0.5,type=float )
    parser.add_argument('--batch_size_lgd',  default= 64,type=int )
    

    arg = parser.parse_args()

    print(arg)
    return arg


def nab(args,result):
    agg = Metric_Aggregator()

    # Turn off all transforms, so that the dataset return PIL.Image.Image object
    result["bd_train"].wrap_img_transform = None
    result["bd_test"].wrap_img_transform = None
    result["clean_test"].wrap_img_transform = None

    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        # print(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        # print(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.random_seed)

    logging.info("===Setup running===")
    
    if args.checkpoint_load == None:
        args.resume = 'False' 
    else :
        args.resume = args.checkpoint_load
    
    if args.dataset == 'cifar10':
        config_file = './utils/defense_utils/dbd/config_z/pretrain/' + 'squareTrigger/' + args.dataset + '/example.yaml'
    else:
        config_file = './utils/defense_utils/dbd/config_z/pretrain/' + 'squareTrigger/imagenet/example.yaml'
    config_ori, inner_dir, config_name = load_config(config_file)
    try:
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    except:
        print('CUDA_VISIBLE_DEVICES is not set. Set GPU=1 now.')
        gpu = 0
 
    logging.info("===Self-Supervise Learning Phase===")

    # Step 1: Train the self-supervised learning model
    information = get_information(args,result,config_ori)

    self_poison_train_loader = information['self_poison_train_loader']
    self_model = information['self_model']
    criterion = information['criterion']
    optimizer = information['optimizer']
    scheduler = information['scheduler']
    resumed_epoch = information['resumed_epoch']

    if os.path.exists(os.getcwd() + args.checkpoint_save + "/self_latest_model.pt"):
        logging.info("Load the latest model from {}".format(os.getcwd() + args.checkpoint_save + "/self_latest_model.pt"))
    else:
        # a.self-supervised learning generates feature extractor
        for epoch in range(args.epoch_self - resumed_epoch):
        
            self_train_result = simclr_train(
                self_model, self_poison_train_loader, criterion, optimizer, logger, False
            )

            if scheduler is not None:
                scheduler.step()
                logging.info(
                    "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
                )

        
            result_self = {"self_train": self_train_result}

            saved_dict = {
                "epoch": epoch + resumed_epoch + 1,
                "result": result_self,
                "model_state_dict": self_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                saved_dict["scheduler_state_dict"] = scheduler.state_dict()

            ckpt_path = os.path.join(os.getcwd() + args.checkpoint_save, "self_latest_model.pt")
            torch.save(saved_dict, ckpt_path)
            logging.info("Save the latest model to {}".format(ckpt_path))
        
    if args.dataset == 'cifar10':
        config_file_semi = './utils/defense_utils/dbd/config_z/semi/' + 'badnets/' + args.dataset + '/example.yaml'
    else:
        config_file_semi = './utils/defense_utils/dbd/config_z/semi/' + 'badnets/imagenet/example.yaml'
 
    finetune_config, finetune_inner_dir, finetune_config_name = load_config(config_file_semi)
    pretrain_config, pretrain_inner_dir, pretrain_config_name = load_config(
        config_file
    )
    ckpt_path = os.path.join(os.getcwd() + args.checkpoint_save, "self_latest_model.pt")
    pretrain_ckpt_path = ckpt_path
    # merge the pretrain and finetune config
    pretrain_config.update(finetune_config)

    pretrain_config['warmup']['criterion']['sce']['num_classes'] = args.num_classes
    pretrain_config['warmup']['num_epochs'] = args.epoch_warmup
   
    backbone = get_network_dbd(args)

    self_model = SelfModel(backbone)
    self_model = self_model.to(args.device)
    # # Load backbone from the pretrained model.
    loc = os.path.join(os.getcwd() + args.checkpoint_save, "self_latest_model.pt")
    load_state(
        self_model, pretrain_config["pretrain_checkpoint"], loc, args.device, logger
    )

    logging.info("\n===Prepare data===")

    result["bd_train"].wrap_img_transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    result["bd_test"].wrap_img_transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    result["clean_test"].wrap_img_transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    
    data_loader = torch.utils.data.DataLoader(result["bd_train"], batch_size=args.batch_size_lgd, num_workers=args.num_workers, shuffle=False)

    # Step 2: Detect suspicious samples:
    logging.info('----------- Network Initialization --------------')
    model_ascent = generate_cls_model(args.model,args.num_classes)
    model_ascent.to(args.device)

    logging.info('finished model init...')
    # initialize optimizer 
    # because the optimizer has parameter nesterov
    args.momentum = 0.9
    args.weight_decay = 5e-4
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

   

    logging.info('----------- Poisoned Sample Detection (LGA) Phase --------------')
    acc_cnt = 0
    all_cnt = 0
    loss_log = 0
    criterion = torch.nn.CrossEntropyLoss()
    if os.path.exists(os.getcwd() + args.checkpoint_save + "/ascent_latest_model.pt"):
        logging.info("Load the latest model from {}".format(os.getcwd() + args.checkpoint_save + "/ascent_latest_model.pt"))
    else:
        for epoch in range(args.epoch_lga):
            model_ascent.train()
            for i, (image, label, *other_info) in enumerate(data_loader):
                image = image.to(args.device)
                label = label.to(args.device)
                logits = model_ascent(image)
                loss = criterion(logits, label)
                loss = (loss - args.gamma).abs() + args.gamma

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc_cnt += (logits.detach().max(1)[1] == label).sum()
                all_cnt += len(label)
                loss_log += loss.detach() * len(label)
            
            train_acc = acc_cnt / all_cnt * 100
            loss = loss_log / all_cnt    
            import math
            lr = 0.5 * (1 + math.cos(math.pi * epoch / (args.epoch_lga + 80))) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logging.info('epoch: {}, train_acc: {:.2f}%, loss: {:.4f}'.format(epoch, train_acc, loss))
        torch.save(model_ascent.state_dict(), os.path.join(os.getcwd() + args.checkpoint_save, "ascent_latest_model.pt"))
    
    model_ascent.load_state_dict(torch.load(os.path.join(os.getcwd() + args.checkpoint_save, "ascent_latest_model.pt")))
    model_ascent.eval()
    # Isolate data
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model_ascent.eval()
    loss_list = []
    idx_list = []
    backdoor_list = []
    with torch.no_grad():
        for i, (image, label, idx, backdoor, ori_label) in enumerate(data_loader):
            image = image.to(args.device)
            label = label.to(args.device)
            out = model_ascent(image)
            loss = criterion(out, label)
            loss_list.append(loss.cpu().squeeze())
            idx_list.append(idx)
            backdoor_list.append(backdoor)
    loss = torch.cat(loss_list)
    idx = torch.cat(idx_list)
    backdoor = torch.cat(backdoor_list)


    # select
    for ratio in (0.01, 0.05, 0.10, 0.2):
        num_iso = int(ratio * len(idx))
        select_isolation = loss.sort()[1][:num_iso]
        idx_iso = idx[select_isolation]

        isolated = torch.zeros(len(idx)).bool()
        isolated.scatter_(0, idx_iso, True)

        attacked_ratio = backdoor[select_isolation].sum() / num_iso
        print("Malign, Ratio {:.2f}, isolated {} among {} samples, with acc: {:.2f}%".format(ratio, num_iso, len(idx), attacked_ratio * 100))
        logging.info("Malign, Ratio {:.2f}, isolated {} among {} samples, with acc: {:.2f}%".format(ratio, num_iso, len(idx), attacked_ratio * 100))
        torch.save(isolated,os.getcwd() + args.checkpoint_save + f"/{args.dataset}_{ratio}_lga")
        
    # Step 3: Relabel
    
    # compute centroids for each class
    class_centroids = [0 for _ in range(args.num_classes)]
    class_n_samples = [0 for _ in range(args.num_classes)]
    temp_iso = torch.load(os.getcwd() + args.checkpoint_save + f"/{args.dataset}_{0.2}_lga")
    
    for i, (image, label, idx, backdoor, ori_label) in enumerate(data_loader):
        image = image.to(args.device)
        label = label.to(args.device)
        out = self_model(image).detach()
        for j in range(len(label)):
            if not temp_iso[idx[j]]:
                class_centroids[label[j]] += out[j]
                class_n_samples[label[j]] += 1
    for i in range(args.num_classes):
        class_centroids[i] /= class_n_samples[i]
        
    # detect suspicious samples
    temp_iso = torch.load(os.getcwd() + args.checkpoint_save + f"/{args.dataset}_{0.1}_lga")
    x_samples = [x for x,y,*other_info in result["bd_train"]]
    y_samples = [y for x,y,*other_info in result["bd_train"]]
    true_label = [other_info[-1] for x,y,*other_info in result["bd_train"]]
    poi_info = [other_info[1] for x,y,*other_info in result["bd_train"]]
    normlization = get_dataset_normalization(args.dataset)
    denormalization = get_dataset_denormalization(normalization=normlization)
    self_model.eval()
    relabel_correct = 0
    detect_correct = 0
    total_detect = 0
    relabel_correct_bd = 0
    relabel_correct_clean = 0
    total_bd = 0
    total_clean = 0
    # oracle case
    # temp_iso = poi_info

    pseudo_label = []
    for i in range(len(x_samples)):
        if not temp_iso[i]:
            pseudo_label.append(y_samples[i])
        else:
            # decide the new label by nearest centroid
            x = x_samples[i].unsqueeze(0).to(args.device)
            out = self_model(x).detach()
            dist = [torch.norm(out - class_centroids[j]) for j in range(args.num_classes)]
            # oracle case 
            # new_label = true_label[i]
            new_label = torch.tensor(dist).argmin().item()
            pseudo_label.append(new_label)

            relabel_correct += (new_label == true_label[i])
            if poi_info[i] == 1:
                relabel_correct_bd += (new_label == true_label[i])
                total_bd += 1
            else:
                relabel_correct_clean += (new_label == true_label[i])
                total_clean += 1
            detect_correct += (poi_info[i] == 1)
            total_detect += 1

    print("Relabel correct: {:.2f}%, Detect correct: {:.2f}%".format(relabel_correct / total_detect * 100, detect_correct / total_detect * 100))
    print("Relabel correct clean: {:.2f}%, Relabel correct bd: {:.2f}%".format(relabel_correct_clean / relabel_correct_clean * 100, relabel_correct_bd / total_bd * 100))
    print('Total detect: ', total_detect)
    logging.info("Relabel correct: {:.2f}%, Detect correct: {:.2f}%".format(relabel_correct / total_detect * 100, detect_correct / total_detect * 100))
    logging.info("Relabel correct clean: {:.2f}%, Relabel correct bd: {:.2f}%".format(relabel_correct_clean / relabel_correct_clean * 100, relabel_correct_bd / total_bd * 100))
    logging.info(f'Total detect: {total_detect}')

    def inject(x):
        x_new = x.clone()
        x_new[:, 0:2, 0:2] = 0.0
        return x_new
    
    def inject_trans(trasform):
        def transform_new(x):
            x = trasform(x)
            x = inject(x)
            return x
        return transform_new
    # Step 4: Retrain

    logging.info('----------- Poisoned Sample Detection (LGA) Phase --------------')

    result["bd_train"].wrap_img_transform = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)

    model = generate_cls_model(args.model,args.num_classes)

    outer_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr,
                                        momentum=args.sgd_momentum,  # 0.9
                                        weight_decay=args.wd,  # 5e-4
                                        )    
    relabel_data_loader = torch.utils.data.DataLoader(result["bd_train"], batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    data_bd_testset = result['bd_test']
    data_bd_testset.wrap_img_transform = inject_trans(test_tran)
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True)

    data_clean_testset = result['clean_test']
    data_clean_testset.wrap_img_transform =  inject_trans(test_tran)
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True)

    clean_test_loss_list = []
    bd_test_loss_list = []
    ra_test_loss_list = []
    test_acc_list = []
    test_asr_list = []
    test_ra_list = []
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    import math
    for epoch in range(args.epochs):
        model.train()
        for images, labels, *other_info in relabel_data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            idx = other_info[0]

            # get pseudo label
            pseudo_label_batch = torch.tensor([pseudo_label[i] for i in idx]).to(args.device)
            isolated_batch = torch.tensor([temp_iso[i] for i in idx]).to(args.device)
            replace = isolated_batch.to(args.device)
            add_trigger = replace & (pseudo_label_batch != labels.to(args.device))
            images[add_trigger,:,:2,:2]=0.0
            labels[replace] = pseudo_label_batch[replace]

            outer_opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            outer_opt.step()

        scheduler.step()
        model.eval()
        clean_test_loss_avg_over_batch, \
        bd_test_loss_avg_over_batch, \
        ra_test_loss_avg_over_batch, \
        test_acc, \
        test_asr, \
        test_ra = eval_step(
            model,
            data_clean_loader,
            data_bd_loader,
            args,
        )

        agg({
            "epoch": epoch,
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
            save_path=os.getcwd()+  f"{args.checkpoint_save}nab_acc_like_metric_plots.png",
            ylabel="percentage",
        )

        general_plot_for_epoch(
            {
                "Test Clean Loss": clean_test_loss_list,
                "Test Backdoor Loss": bd_test_loss_list,
                "Test RA Loss": ra_test_loss_list,
            },
            save_path=os.getcwd()+f"{args.checkpoint_save}nab_loss_metric_plots.png",
            ylabel="percentage",
        )

        agg.to_dataframe().to_csv(os.getcwd()+f"{args.checkpoint_save}nab_df.csv")

    agg.summary().to_csv(os.getcwd()+f"{args.checkpoint_save}nab_df_summary.csv")
    
    result = {}
    result['model'] = model
    save_defense_result(
        model_name=args.model,
        num_classes=args.num_classes,
        model=model.cpu().state_dict(),
        save_path=os.getcwd()+args.checkpoint_save,
    )
    return result

def eval_step(
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
        device=args.device,
        verbose=0,
    )
    clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
    test_acc = clean_metrics['test_acc']
    bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
        netC,
        bd_test_dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        non_blocking=args.non_blocking,
        device=args.device,
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
        device=args.device,
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
    
   ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    # args.result_file = 'badnet_demo'
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/defense/nab/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = args.checkpoint_save + 'log/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
        # args.log = save_path + '/saved/nab/'
        # if not (os.path.exists(os.getcwd() + args.log)):
        #     os.makedirs(os.getcwd() + args.log) 
    args.save_path = save_path
    
    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ### 3. nab defense:
    print("Continue training...")
    result_defense = nab(args,result)


    ### 4. test the result and get ASR, ACC, RC
    # resume transfroms
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    result["bd_test"].wrap_img_transform = tran
    result["clean_test"].wrap_img_transform = tran

    result_defense['model'].eval()
    result_defense['model'].to(args.device) 
    data_bd_testset = result['bd_test']
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    asr_acc = 0
    for i, (inputs,labels, *other_info) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_testset)

    data_clean_testset = result['clean_test']
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    clean_acc = 0
    for i, (inputs,labels, *other_info) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)
    clean_acc = clean_acc/len(data_clean_testset)

    robust_acc = 0
    for i, (inputs,labels, original_index, poison_indicator, original_targets) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        original_targets = original_targets.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        robust_acc += torch.sum(pre_label == original_targets)
    robust_acc = robust_acc/len(data_bd_testset)

    print('ACC: ', clean_acc)
    print('ASR: ', asr_acc)
    print('RA: ', robust_acc)

    if not (os.path.exists(os.getcwd() + f'{save_path}/nab/')):
        os.makedirs(os.getcwd() + f'{save_path}/nab/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    f'./{save_path}/nab/defense_result.pt'
    )
    # test_acc,test_asr,test_ra
    final_result = {'model_name':args.model, 'test_acc':clean_acc.item(), 'test_asr':asr_acc.item(), 'test_ra':robust_acc.item()}
    # to csv 
    import pandas as pd
    df = pd.DataFrame(final_result, index=[0])
    df.to_csv(f'./{save_path}/nab/nab_df_summary.csv', index=False)
