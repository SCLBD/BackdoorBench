import argparse
import shutil
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from data.dataset import PoisonLabelDataset
from model.model import LinearModel
from model.utils import (
    get_criterion,
    get_network,
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.setup import (
    get_logger,
    load_config,
    get_saved_dir,
    get_storage_dir,
    set_seed,
)
from utils.trainer.log import result2csv
from utils.trainer.supervise import poison_train, test


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/supervise/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )
    parser.add_argument("--amp", default=False, action="store_true")
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
    logger = get_logger(args.log_dir, "supervise.log", args.resume, gpu == 0)
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
    test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    test_transform = {
        "pre": pre_transform,
        "primary": test_primary_transform,
        "remaining": test_remaining_transform,
    }
    logger.info("Test transformations:\n {}".format(test_transform))

    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    clean_train_data = get_dataset(
        config["dataset_dir"], train_transform, prefetch=config["prefetch"]
    )
    poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
    poison_idx_path = os.path.join(args.saved_dir, "poison_idx.npy")
    np.save(poison_idx_path, poison_train_idx)
    logger.info("Save poisoned index to {}".format(poison_idx_path))
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )
    poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )
    if args.distributed:
        poison_train_sampler = DistributedSampler(poison_train_data)
        batch_size = int(config["loader"]["batch_size"] / ngpus_per_node)
        num_workers = config["loader"]["num_workers"]
        poison_train_loader = get_loader(
            poison_train_data,
            batch_size=batch_size,
            sampler=poison_train_sampler,
            num_workers=num_workers,
        )
    else:
        poison_train_sampler = None
        poison_train_loader = get_loader(
            poison_train_data, config["loader"], shuffle=True
        )

    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])

    logger.info("\n===Setup training===")
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)
    if args.distributed:
        linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])
    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {}".format(criterion))
    optimizer = get_optimizer(linear_model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    resumed_epoch, best_acc, best_epoch = load_state(
        linear_model,
        args.resume,
        args.ckpt_dir,
        gpu,
        logger,
        optimizer,
        scheduler,
        is_best=True,
    )

    for epoch in range(config["num_epochs"] - resumed_epoch):
        if args.distributed:
            poison_train_sampler.set_epoch(epoch)
        logger.info(
            "===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, config["num_epochs"])
        )
        logger.info("Poison training...")
        poison_train_result = poison_train(
            linear_model,
            poison_train_loader,
            criterion,
            optimizer,
            logger,
            amp=args.amp,
        )
        logger.info("Test model on clean data...")
        clean_test_result = test(linear_model, clean_test_loader, criterion, logger)
        logger.info("Test model on poison data...")
        poison_test_result = test(linear_model, poison_test_loader, criterion, logger)

        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )

        # Save result and checkpoint.
        if not args.distributed or (args.distributed and gpu == 0):
            result = {
                "poison_train": poison_train_result,
                "clean_test": clean_test_result,
                "poison_test": poison_test_result,
            }
            result2csv(result, args.log_dir)

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

            is_best = False
            if clean_test_result["acc"] > best_acc:
                is_best = True
                best_acc = clean_test_result["acc"]
                best_epoch = epoch + resumed_epoch + 1
            logger.info(
                "Best test accuaracy {} in epoch {}".format(best_acc, best_epoch)
            )
            if is_best:
                ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
                torch.save(saved_dict, ckpt_path)
                logger.info("Save the best model to {}".format(ckpt_path))
            ckpt_path = os.path.join(args.ckpt_dir, "latest_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the latest model to {}".format(ckpt_path))


if __name__ == "__main__":
    main()
