'''
@misc{huang2022backdoor,
  title={Backdoor defense via decoupling the training process},
  author={Huang, Kunzhe and Li, Yiming and Wu, Baoyuan and Qin, Zhan and Ren, Kui},
  year={2022},
  publisher={ICLR}
}

code : https://github.com/SCLBD/DBD
'''
import logging
import time
import argparse
import shutil
import os

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchaudio import transforms

import yaml

from torch.utils.data.distributed import DistributedSampler


from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
    get_semi_idx,
)
from data.dataset import PoisonLabelDataset, SelfPoisonDataset, MixMatchDataset
from utils_db.box import get_information
from model.model import SelfModel, LinearModel
from model.utils import (
    load_state,
    get_criterion,
    get_network,
    get_optimizer,
    get_saved_epoch,
    get_scheduler,
)
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils_db.setup import (
    load_config,
    get_logger,
    get_saved_dir,
    get_storage_dir,
    set_seed,
)
from utils_db.trainer.log import result2csv
from utils_db.trainer.simclr import simclr_train




from utils_db.trainer.semi import mixmatch_train
from utils_db.trainer.simclr import linear_test, poison_linear_record, poison_linear_train

def get_args():
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
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    ####K_arm
    parser.add_argument('--gamma',type=float,help='gamma for pre-screening') 
    parser.add_argument('--global_theta',type=float,help='theta for global trigger pre-screening') 
    parser.add_argument('--local_theta',type=float,help='theta for label-specific trigger pre-screening') 
    
    parser.add_argument('--sym_check',type=bool,help='If using sym check') 
    parser.add_argument('--global_det_bound',type=int,help='global bound to decide whether the model is trojan or not') 
    parser.add_argument('--local_det_bound',type=int,help='local bound to decide whether the model is trojan or not') 
    parser.add_argument('--ratio_det_bound',type=int,help='ratio bound to decide whether the model is trojan or not') 
    
    parser.add_argument('--regularization',type=str )
    parser.add_argument('--init_cost',type=float )
    parser.add_argument('--step',type=int )
    parser.add_argument('--rounds',type=int )
    parser.add_argument('--lr_re',type=float )
    parser.add_argument('--patience',type=int )
    parser.add_argument('--attack_succ_threshold',type=float )
    parser.add_argument('--single_color_opt',type=bool ) 
    parser.add_argument('--warmup_rounds',type=int )
    parser.add_argument('--epsilon_for_bandits',type=float )
    parser.add_argument('--epsilon',type=float )
    parser.add_argument('--beta',type=float,help='beta in the objective function') 
    parser.add_argument('--cost_multiplier',type=float )
    parser.add_argument('--early_stop',type=bool )
    parser.add_argument('--early_stop_threshold',type=float )
    parser.add_argument('--early_stop_patience',type=int )
    parser.add_argument('--reset_cost_to_zero',type=bool )
    parser.add_argument('--central_init',type=bool,help='strategy for initalization') 

    arg = parser.parse_args()

    print(arg)
    return arg


def DBD(args,result,config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    if args.log is not None & args.log != '':
        fileHandler = logging.FileHandler('./log' + '/' + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler('./log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    print("===Setup running===")
    # parser = argparse.ArgumentParser()
    #parser.add_argument("--config", default="./config/pretrain/example.yaml")
    #parser.add_argument("--gpu", default="0", type=str)
    # parser.add_argument(
    #     "--resume",
    #     default="",
    #     type=str,
    #     help="checkpoint name (empty string means the latest checkpoint)\
    #         or False (means training from scratch).",
    # )
    args.resume = args.checkpoint_load
    # parser.add_argument("--amp", default=False, action="store_true")
    # parser.add_argument("--num_stage_epochs", default=100, type=int)
    # parser.add_argument("--min_interval", default=20, type=int)
    # parser.add_argument("--max_interval", default=100, type=int)
    # parser.add_argument(
    #     "--world-size",
    #     default=1,
    #     type=int,
    #     help="number of nodes for distributed training",
    # )
    # parser.add_argument(
    #     "--rank", default=0, type=int, help="node rank for distributed training"
    # )
    # parser.add_argument(
    #     "--dist-port",
    #     default="23456",
    #     type=str,
    #     help="port used to set up distributed training",
    # )
    # args = parser.parse_args()
    
    ####需要修改trigger类型
    config_file = './defense/DBD/config_z/pretrain/' + 'signalTrigger/' + args.dataset + '/example.yaml'
    config_ori, inner_dir, config_name = load_config(config_file)


    args.saved_dir, args.log_dir = get_saved_dir(
        config_ori, inner_dir, config_name, args.resume
    )
    # shutil.copy2(args.config, args.saved_dir)
    # args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
    #     config, inner_dir, config_name, args.resume
    # )
    # shutil.copy2(args.config, args.storage_dir)
    # set_seed(**config["seed"])

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # ngpus_per_node = torch.cuda.device_count()
    # if ngpus_per_node > 1:
    #     args.distributed = True
    # else:
    #     args.distributed = False
    # if args.distributed:
    #     args.world_size = ngpus_per_node * args.world_size
    #     print("Distributed training on GPUs: {}.".format(args.gpu))
    #     mp.spawn(
    #         main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config),
    #     )
    # else:
    #     print("Training on a single GPU: {}.".format(args.gpu))
    gpu = 0
    logger = get_logger(args.log, "pretrain.log", args.resume, gpu == 0)
    torch.cuda.set_device(gpu)
    # if args.distributed:
    #     args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(
    #         backend="nccl",
    #         init_method="tcp://127.0.0.1:{}".format(args.dist_port),
    #         world_size=args.world_size,
    #         rank=args.rank,
    #     )
    #     logger.warning("Only log rank 0 in distributed training!")
    # if args.amp:
    #     logger.info("Turn on PyTorch native automatic mixed precision.")

    logger.info("===Prepare data===")
    #bd_config = config["backdoor"]
    #logger.info("Load backdoor config:\n{}".format(bd_config))
    #bd_transform = get_bd_transform(bd_config)
    #target_label = bd_config["target_label"]
    #poison_ratio = bd_config["poison_ratio"]

    information = get_information(args,result,config_ori)
   
    # saved_epoch = get_saved_epoch(
    #     config["num_epochs"],
    #     args.num_stage_epochs,
    #     args.min_interval,
    #     args.max_interval,
    # )
    # logger.info("Set saved epoch to {}".format(saved_epoch))

    self_poison_train_loader = information['self_poison_train_loader']
    self_model = information['self_model']
    criterion = information['criterion']
    optimizer = information['optimizer']
    scheduler = information['scheduler']
    resumed_epoch = information['resumed_epoch']

    for epoch in range(args.epoch - resumed_epoch):
        # if args.distributed:
        #     self_poison_train_sampler.set_epoch(epoch)
        # logger.info(
        #     "===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, config["num_epochs"])
        # )
        # logger.info("SimCLR training...")
        self_train_result = simclr_train(
            self_model, self_poison_train_loader, criterion, optimizer, logger, False
        )

        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )

        # Save result and checkpoint.
        if not args.distributed or (args.distributed and gpu == 0):
            result = {"self_train": self_train_result}
            result2csv(result, args.log_dir)

            saved_dict = {
                "epoch": epoch + resumed_epoch + 1,
                "result": result,
                "model_state_dict": self_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                saved_dict["scheduler_state_dict"] = scheduler.state_dict()

            ckpt_path = os.path.join(args.ckpt_dir, "latest_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the latest model to {}".format(ckpt_path))
            
            ckpt_path = os.path.join(
                args.checkpoint_save, "epoch{}.pt".format(epoch + resumed_epoch + 1)
            )
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the model in saved epoch to {}".format(ckpt_path))

    ####self
    # parser = argparse.ArgumentParser()
    #parser.add_argument("--config", default="./config/pretrain/example.yaml")
    #parser.add_argument("--gpu", default="0", type=str)
    # parser.add_argument(
    #     "--resume",
    #     default="",
    #     type=str,
    #     help="checkpoint name (empty string means the latest checkpoint)\
    #         or False (means training from scratch).",
    # )
    # args.resume = args.checkpoint_load
    # parser.add_argument("--amp", default=False, action="store_true")
    # parser.add_argument("--num_stage_epochs", default=100, type=int)
    # parser.add_argument("--min_interval", default=20, type=int)
    # parser.add_argument("--max_interval", default=100, type=int)
    # parser.add_argument(
    #     "--world-size",
    #     default=1,
    #     type=int,
    #     help="number of nodes for distributed training",
    # )
    # parser.add_argument(
    #     "--rank", default=0, type=int, help="node rank for distributed training"
    # )
    # parser.add_argument(
    #     "--dist-port",
    #     default="23456",
    #     type=str,
    #     help="port used to set up distributed training",
    # )
    # args = parser.parse_args()
    
    ####需要修改trigger类型
    config_file_semi = './config_z/semi/' + 'signalTrigger/' + args.dataset + '/example.yaml'

    # config, inner_dir, config_name = load_config(args.config)
    # args.saved_dir, args.log_dir = get_saved_dir(
    #     config, inner_dir, config_name, args.resume
    # )
    # shutil.copy2(args.config, args.saved_dir)
    # args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
    #     config, inner_dir, config_name, args.resume
    # )
    # shutil.copy2(args.config, args.storage_dir)
    # set_seed(**config["seed"])

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #####sim supervised
    finetune_config, finetune_inner_dir, finetune_config_name = load_config(config_file_semi)
    pretrain_config, pretrain_inner_dir, pretrain_config_name = load_config(
        finetune_config["pretrain_config_path"]
    )
    # pretrain_saved_dir, _ = get_saved_dir(
    #     pretrain_config, pretrain_inner_dir, pretrain_config_name
    # )
    _, pretrain_ckpt_dir, _ = get_storage_dir(
        pretrain_config, pretrain_inner_dir, pretrain_config_name
    )
    # merge the pretrain and finetune config
    pretrain_config.update(finetune_config)

    ####替换某些参数
    pretrain_config['warmup']['criterion']['sce']['num_classes'] = args.num_classes
    pretrain_config['warmup']['num_epochs'] = args.epoch_warmup

    #config = pretrain_config
    # saved_dir, log_dir = get_saved_dir(
    #     config, finetune_inner_dir, finetune_config_name, args.resume
    # )
    # shutil.copy2(args.config, saved_dir)
    # storage_dir, ckpt_dir, record_dir = get_storage_dir(
    #     config, finetune_inner_dir, finetune_config_name, args.resume,
    # )
    # shutil.copy2(args.config, storage_dir)
    # logger = get_logger(log_dir, "finetune.log", args.resume)
    # set_seed(**config["seed"])
    # logger.info("Load finetune config from: {}".format(args.config))
    # logger.info(
    #     "Load pretrain config from: {}".format(finetune_config["pretrain_config_path"])
    # )

    logger.info("\n===Prepare data===")
    # bd_config = config["backdoor"]
    # logger.info("Load backdoor config:\n{}".format(bd_config))
    # bd_transform = get_bd_transform(bd_config)
    # target_label = bd_config["target_label"]
    # poison_ratio = bd_config["poison_ratio"]

    # pre_transform = get_transform(config["transform"]["pre"])
    # train_primary_transform = get_transform(config["transform"]["train"]["primary"])
    # train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
    # train_transform = {
    #     "pre": pre_transform,
    #     "primary": train_primary_transform,
    #     "remaining": train_remaining_transform,
    # }
    # logger.info("Training transformations:\n {}".format(train_transform))
    # test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    # test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    # test_transform = {
    #     "pre": pre_transform,
    #     "primary": test_primary_transform,
    #     "remaining": test_remaining_transform,
    # }
    # logger.info("Test transformations:\n {}".format(test_transform))

    # logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    # clean_train_data = get_dataset(
    #     config["dataset_dir"], train_transform, prefetch=config["prefetch"]
    # )
    # # Load poisoned training index from pretrain.
    # poison_idx_path = os.path.join(pretrain_saved_dir, "poison_idx.npy")
    # poison_train_idx = np.load(poison_idx_path)
    # poison_train_data = PoisonLabelDataset(
    #     clean_train_data, bd_transform, poison_train_idx, target_label
    # )
    # clean_test_data = get_dataset(
    #     config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    # )
    # poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    # poison_test_data = PoisonLabelDataset(
    #     clean_test_data, bd_transform, poison_test_idx, target_label
    # )

    train_primary_transform = get_transform(pretrain_config["transform"]["train"]["primary"])
    train_remaining_transform = get_transform(pretrain_config["transform"]["train"]["remaining"])
    train_transform = transforms.Compose([train_primary_transform,train_remaining_transform])
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy()))
    y = result['bd_train']['y']
    data_set = torch.utils.data.TensorDataset(x,y)
    dataset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=train_transform,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    poison_train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    poison_eval_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)
    
    test_primary_transform = get_transform(pretrain_config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(pretrain_config["transform"]["test"]["remaining"])
    test_transform = transforms.Compose([test_primary_transform,test_remaining_transform])
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
    y = result['bd_train']['y']
    data_set_te_bd = torch.utils.data.TensorDataset(x,y)
    dataset_te_bd = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set_te_bd,
        poison_idx=np.zeros(len(data_set_te_bd)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=test_transform,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    poison_test_loader = torch.utils.data.DataLoader(dataset_te_bd, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)

    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
    y = result['clean_train']['y']
    data_set_te_cl = torch.utils.data.TensorDataset(x,y)
    dataset_te_cl = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set_te_cl,
        poison_idx=np.zeros(len(data_set_te_cl)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=test_transform,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    clean_test_loader = torch.utils.data.DataLoader(dataset_te_cl, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)

    # logger.info("\n===Setup training===")
    # gpu = int(args.gpu)
    # torch.cuda.set_device(gpu)
    # logger.info("Set gpu to: {}".format(gpu))
    backbone = get_network(pretrain_config["network"])
    # logger.info("Create network: {}".format(config["network"]))
    self_model = SelfModel(backbone)
    self_model = self_model.to(args.device)
    # Load backbone from the pretrained model.
    load_state(
        self_model, pretrain_config["pretrain_checkpoint"], pretrain_ckpt_dir, args.device, logger
    )
    linear_model = LinearModel(backbone, backbone.feature_dim, args.num_classes)
    linear_model.linear.to(args.device)
    warmup_criterion = get_criterion(pretrain_config["warmup"]["criterion"])
    logger.info("Create criterion: {} for warmup".format(warmup_criterion))
    warmup_criterion = warmup_criterion.to(args.device)
    semi_criterion = get_criterion(pretrain_config["semi"]["criterion"])
    semi_criterion = semi_criterion.to(args.device)
    logger.info("Create criterion: {} for semi-training".format(semi_criterion))
    optimizer = get_optimizer(linear_model, pretrain_config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    scheduler = get_scheduler(optimizer, pretrain_config["lr_scheduler"])
    logger.info("Create learning rete scheduler: {}".format(pretrain_config["lr_scheduler"]))
    if args.checkpoint_load == '' or args.checkpoint_load is None:
        resume = 'False'
    resumed_epoch, best_acc, best_epoch = load_state(
        linear_model,
        resume,
        args.checkpoint_load,
        gpu,
        logger,
        optimizer,
        scheduler,
        is_best=True,
    )

    num_epochs = args.epoch_warmup + args.epoch
    for epoch in range(num_epochs - resumed_epoch):
        logger.info("===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, num_epochs))
        if (epoch + resumed_epoch + 1) <= args.epoch_warmup:
            logger.info("Poisoned linear warmup...")
            poison_train_result = poison_linear_train(
                linear_model, poison_train_loader, warmup_criterion, optimizer, logger,
            )
        else:
            record_list = poison_linear_record(
                linear_model, poison_eval_loader, warmup_criterion
            )
            logger.info("Mining clean data from poisoned dataset...")
            semi_idx = get_semi_idx(record_list, args.epsilon, logger)
            xdata = MixMatchDataset(dataset_te_cl, semi_idx, labeled=True)
            udata = MixMatchDataset(dataset_te_cl, semi_idx, labeled=False)
            xloader = get_loader(
                xdata, pretrain_config["semi"]["loader"], shuffle=True, drop_last=True
            )
            uloader = get_loader(
                udata, pretrain_config["semi"]["loader"], shuffle=True, drop_last=True
            )
            logger.info("MixMatch training...")
            poison_train_result = mixmatch_train(
                linear_model,
                xloader,
                uloader,
                semi_criterion,
                optimizer,
                epoch,
                logger,
                **config["semi"]["mixmatch"]
            )
        logger.info("Test model on clean data...")
        clean_test_result = linear_test(
            linear_model, clean_test_loader, warmup_criterion, logger
        )
        logger.info("Test model on poison data...")
        poison_test_result = linear_test(
            linear_model, poison_test_loader, warmup_criterion, logger
        )
        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )
        result = {
            "poison_train": poison_train_result,
            "poison_test": poison_test_result,
            "clean_test": clean_test_result,
        }
        result2csv(result, args.log)

        is_best = False
        if clean_test_result["acc"] > best_acc:
            is_best = True
            best_acc = clean_test_result["acc"]
            best_epoch = epoch + resumed_epoch + 1
        logger.info("Best test accuaracy {} in epoch {}".format(best_acc, best_epoch))

        saved_dict = {
            "epoch": epoch + resumed_epoch + 1,
            "result": result,
            "model_state_dict": linear_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        if scheduler is not None:
            saved_dict["scheduler_state_dict"] = scheduler.state_dict()

        if is_best:
            ckpt_path = os.path.join(args.checkpoint_save, "best_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the best model to {}".format(ckpt_path))
        ckpt_path = os.path.join(args.checkpoint_save, "latest_model.pt")
        torch.save(saved_dict, ckpt_path)
        logger.info("Save the latest model to {}".format(ckpt_path))

    result = {}
    result['model'] = linear_model
    return result

if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/DBD/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    args.checkpoint_save = os.getcwd() + '/record/defence/ac/' + args.dataset + '.tar'
    args.log = 'saved/log/log_' + args.dataset + '.txt'

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    args.save_path = save_path
    result = torch.load(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = DBD(args,result,config)
        
        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
        y = result['bd_test']['y']
        data_bd_test = torch.utils.data.TensorDataset(x,y)
        data_bd_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_bd_test,
            poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        asr_acc = 0
        for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
        y = result['clean_test']['y']
        data_clean_test = torch.utils.data.TensorDataset(x,y)
        data_clean_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_clean_test,
            poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        clean_acc = 0
        for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)


        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc
        },
        f'{save_path}/defense_result.pt'
        )
    else:
        print("There is no target model")
       