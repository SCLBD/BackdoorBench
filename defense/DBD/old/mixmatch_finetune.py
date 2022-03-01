import argparse
import os
import shutil

import numpy as np
import torch

from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
    get_semi_idx,
)
from data.dataset import PoisonLabelDataset, MixMatchDataset
from model.model import SelfModel, LinearModel
from model.utils import (
    get_criterion,
    get_network,
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.trainer.log import result2csv
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.semi import mixmatch_train
from utils.trainer.simclr import linear_test, poison_linear_record, poison_linear_train


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/defense/semi/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )
    args = parser.parse_args()

    finetune_config, finetune_inner_dir, finetune_config_name = load_config(args.config)
    pretrain_config, pretrain_inner_dir, pretrain_config_name = load_config(
        finetune_config["pretrain_config_path"]
    )
    pretrain_saved_dir, _ = get_saved_dir(
        pretrain_config, pretrain_inner_dir, pretrain_config_name
    )
    _, pretrain_ckpt_dir, _ = get_storage_dir(
        pretrain_config, pretrain_inner_dir, pretrain_config_name
    )
    # merge the pretrain and finetune config
    pretrain_config.update(finetune_config)
    config = pretrain_config
    saved_dir, log_dir = get_saved_dir(
        config, finetune_inner_dir, finetune_config_name, args.resume
    )
    shutil.copy2(args.config, saved_dir)
    storage_dir, ckpt_dir, record_dir = get_storage_dir(
        config, finetune_inner_dir, finetune_config_name, args.resume,
    )
    shutil.copy2(args.config, storage_dir)
    logger = get_logger(log_dir, "finetune.log", args.resume)
    set_seed(**config["seed"])
    logger.info("Load finetune config from: {}".format(args.config))
    logger.info(
        "Load pretrain config from: {}".format(finetune_config["pretrain_config_path"])
    )

    logger.info("\n===Prepare data===")
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
    # Load poisoned training index from pretrain.
    poison_idx_path = os.path.join(pretrain_saved_dir, "poison_idx.npy")
    poison_train_idx = np.load(poison_idx_path)
    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label
    )
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )
    poison_train_loader = get_loader(
        poison_train_data, config["warmup"]["loader"], shuffle=True
    )
    poison_eval_loader = get_loader(poison_train_data, config["warmup"]["loader"])
    clean_test_loader = get_loader(clean_test_data, config["warmup"]["loader"])
    poison_test_loader = get_loader(poison_test_data, config["warmup"]["loader"])

    logger.info("\n===Setup training===")
    gpu = int(args.gpu)
    torch.cuda.set_device(gpu)
    logger.info("Set gpu to: {}".format(gpu))
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    self_model = SelfModel(backbone)
    self_model = self_model.cuda(gpu)
    # Load backbone from the pretrained model.
    load_state(
        self_model, config["pretrain_checkpoint"], pretrain_ckpt_dir, gpu, logger
    )
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model.linear.cuda(gpu)
    warmup_criterion = get_criterion(config["warmup"]["criterion"])
    logger.info("Create criterion: {} for warmup".format(warmup_criterion))
    warmup_criterion = warmup_criterion.cuda(gpu)
    semi_criterion = get_criterion(config["semi"]["criterion"])
    semi_criterion = semi_criterion.cuda(gpu)
    logger.info("Create criterion: {} for semi-training".format(semi_criterion))
    optimizer = get_optimizer(linear_model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create learning rete scheduler: {}".format(config["lr_scheduler"]))
    resumed_epoch, best_acc, best_epoch = load_state(
        linear_model,
        args.resume,
        ckpt_dir,
        gpu,
        logger,
        optimizer,
        scheduler,
        is_best=True,
    )

    num_epochs = config["warmup"]["num_epochs"] + config["semi"]["num_epochs"]
    for epoch in range(num_epochs - resumed_epoch):
        logger.info("===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, num_epochs))
        if (epoch + resumed_epoch + 1) <= config["warmup"]["num_epochs"]:
            logger.info("Poisoned linear warmup...")
            poison_train_result = poison_linear_train(
                linear_model, poison_train_loader, warmup_criterion, optimizer, logger,
            )
        else:
            record_list = poison_linear_record(
                linear_model, poison_eval_loader, warmup_criterion
            )
            logger.info("Mining clean data from poisoned dataset...")
            semi_idx = get_semi_idx(record_list, config["semi"]["epsilon"], logger)
            xdata = MixMatchDataset(poison_train_data, semi_idx, labeled=True)
            udata = MixMatchDataset(poison_train_data, semi_idx, labeled=False)
            xloader = get_loader(
                xdata, config["semi"]["loader"], shuffle=True, drop_last=True
            )
            uloader = get_loader(
                udata, config["semi"]["loader"], shuffle=True, drop_last=True
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
        result2csv(result, log_dir)

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
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the best model to {}".format(ckpt_path))
        ckpt_path = os.path.join(ckpt_dir, "latest_model.pt")
        torch.save(saved_dict, ckpt_path)
        logger.info("Save the latest model to {}".format(ckpt_path))


if __name__ == "__main__":
    main()
