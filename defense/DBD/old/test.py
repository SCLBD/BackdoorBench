import argparse
import logging

import torch

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
    load_state,
)
from utils.setup import (
    load_config,
    set_seed,
)
from utils.trainer.supervise import test


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/supervise/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--ckpt-dir", default="checkpoint", type=str, help="checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, help="checkpoint name",
    )
    args = parser.parse_args()

    config, _, _ = load_config(args.config)
    set_seed(**config["seed"])

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    logger.info("===Prepare data===")
    bd_config = config["backdoor"]
    logger.info("Load backdoor config:\n{}".format(bd_config))
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]

    pre_transform = get_transform(config["transform"]["pre"])
    test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    test_transform = {
        "pre": pre_transform,
        "primary": test_primary_transform,
        "remaining": test_remaining_transform,
    }
    logger.info("Test transformations:\n {}".format(test_transform))

    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"]
    )
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )

    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])

    logger.info("\n===Setup training===")
    gpu = int(args.gpu)
    torch.cuda.set_device(gpu)
    logger.info("Set gpu to: {}".format(gpu))
    backbone = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))

    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model = linear_model.cuda(gpu)
    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {}".format(criterion))
    load_state(linear_model, args.resume, args.ckpt_dir, gpu, logger)

    logger.info("Test model on clean data...")
    test(linear_model, clean_test_loader, criterion, logger)
    logger.info("Test model on poison data...")
    test(linear_model, poison_test_loader, criterion, logger)


if __name__ == "__main__":
    main()
