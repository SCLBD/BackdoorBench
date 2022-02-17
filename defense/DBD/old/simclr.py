import argparse
import shutil
import os

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from data.dataset import PoisonLabelDataset, SelfPoisonDataset
from model.model import SelfModel
from model.utils import (
    load_state,
    get_criterion,
    get_network,
    get_optimizer,
    get_saved_epoch,
    get_scheduler,
)
from utils.setup import (
    load_config,
    get_logger,
    get_saved_dir,
    get_storage_dir,
    set_seed,
)
from utils.trainer.log import result2csv
from utils.trainer.simclr import simclr_train


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/pretrain/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--num_stage_epochs", default=100, type=int)
    parser.add_argument("--min_interval", default=20, type=int)
    parser.add_argument("--max_interval", default=100, type=int)
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-port",
        default="23456",
        type=str,
        help="port used to set up distributed training",
    )
    args = parser.parse_args()

    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.saved_dir)
    args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.storage_dir)
    set_seed(**config["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.distributed = True
    else:
        args.distributed = False
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("Distributed training on GPUs: {}.".format(args.gpu))
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config),
        )
    else:
        print("Training on a single GPU: {}.".format(args.gpu))
        main_worker(0, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    logger = get_logger(args.log_dir, "pretrain.log", args.resume, gpu == 0)
    torch.cuda.set_device(gpu)
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(args.dist_port),
            world_size=args.world_size,
            rank=args.rank,
        )
        logger.warning("Only log rank 0 in distributed training!")
    if args.amp:
        logger.info("Turn on PyTorch native automatic mixed precision.")

    logger.info("===Prepare data===")
    bd_config = config["backdoor"]
    logger.info("Load backdoor config:\n{}".format(bd_config))
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]
    poison_ratio = bd_config["poison_ratio"]

    pre_transform = get_transform(config["transform"]["pre"])
    train_primary_transform = get_transform(config["transform"]["train"]["primary"])
    train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
    train_transform = {
        "pre": pre_transform,
        "primary": train_primary_transform,
        "remaining": train_remaining_transform,
    }
    logger.info("Training transformations:\n {}".format(train_transform))
    aug_primary_transform = get_transform(config["transform"]["aug"]["primary"])
    aug_remaining_transform = get_transform(config["transform"]["aug"]["remaining"])
    aug_transform = {
        "pre": pre_transform,
        "primary": aug_primary_transform,
        "remaining": aug_remaining_transform,
    }
    logger.info("Augmented transformations:\n {}".format(aug_transform))

    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    clean_train_data = get_dataset(config["dataset_dir"], train_transform)
    poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
    poison_idx_path = os.path.join(args.saved_dir, "poison_idx.npy")
    np.save(poison_idx_path, poison_train_idx)
    logger.info("Save poisoned index to {}".format(poison_idx_path))
    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    self_poison_train_data = SelfPoisonDataset(poison_train_data, aug_transform)
    if args.distributed:
        self_poison_train_sampler = DistributedSampler(self_poison_train_data)
        batch_size = int(config["loader"]["batch_size"] / ngpus_per_node)
        num_workers = config["loader"]["num_workers"]
        self_poison_train_loader = get_loader(
            self_poison_train_data,
            batch_size=batch_size,
            sampler=self_poison_train_sampler,
            num_workers=num_workers,
        )
    else:
        self_poison_train_sampler = None
        self_poison_train_loader = get_loader(
            self_poison_train_data, config["loader"], shuffle=True
        )

    logger.info("\n===Setup training===")
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    self_model = SelfModel(backbone)
    self_model = self_model.cuda(gpu)
    if args.distributed:
        # Convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if config["sync_bn"]:
            self_model = nn.SyncBatchNorm.convert_sync_batchnorm(self_model)
            logger.info("Turn on synchronized batch normalization in ddp.")
        self_model = nn.parallel.DistributedDataParallel(self_model, device_ids=[gpu])
    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {}".format(criterion))
    optimizer = get_optimizer(self_model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    resumed_epoch = load_state(
        self_model, args.resume, args.ckpt_dir, gpu, logger, optimizer, scheduler,
    )
    saved_epoch = get_saved_epoch(
        config["num_epochs"],
        args.num_stage_epochs,
        args.min_interval,
        args.max_interval,
    )
    logger.info("Set saved epoch to {}".format(saved_epoch))

    for epoch in range(config["num_epochs"] - resumed_epoch):
        if args.distributed:
            self_poison_train_sampler.set_epoch(epoch)
        logger.info(
            "===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, config["num_epochs"])
        )
        logger.info("SimCLR training...")
        self_train_result = simclr_train(
            self_model, self_poison_train_loader, criterion, optimizer, logger, args.amp
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
            if (epoch + resumed_epoch + 1) in saved_epoch:
                ckpt_path = os.path.join(
                    args.ckpt_dir, "epoch{}.pt".format(epoch + resumed_epoch + 1)
                )
                torch.save(saved_dict, ckpt_path)
                logger.info("Save the model in saved epoch to {}".format(ckpt_path))


if __name__ == "__main__":
    main()
