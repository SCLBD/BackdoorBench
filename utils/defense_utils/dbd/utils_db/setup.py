import logging
import random
import os
import platform
import shutil
import sys
import time

import numpy as np
import torch
import yaml


def load_config(config_path):
    """Load config file from `config_path`.

    Args:
        config_path (str): Configuration file path, which must be in `config` dir, e.g.,
            `./config/inner_dir/example.yaml` and `config/inner_dir/example`.
    
    Returns:
        config (dict): Configuration dict.
        inner_dir (str): Directory between `config/` and configuration file. If `config_path`
           doesn't contain `inner_dir`, return empty string.
        config_name (str): Configuration filename.
    """
    assert os.path.exists(config_path)
    config_hierarchy = config_path.split("/")
    # if config_hierarchy[0] != ".":
    #     if config_hierarchy[0] != "config":
    #         raise RuntimeError(
    #             "Configuration file {} must be in config dir".format(config_path)
    #         )
    #     if len(config_hierarchy) > 2:
    #         inner_dir = os.path.join(*config_hierarchy[1:-1])
    #     else:
    #         inner_dir = ""
    # else:
    #     # if config_hierarchy[1] != "config_z":
    #     #     raise RuntimeError(
    #     #         "Configuration file {} must be in config dir".format(config_path)
    #     #     )
    #     if len(config_hierarchy) > 3:
    #         inner_dir = os.path.join(*config_hierarchy[2:-1])
    #     else:
    #         inner_dir = ""
    print("Load configuration file from {}:".format(config_path))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_name = config_hierarchy[-1].split(".yaml")[0]

    return config


def get_saved_dir(config, inner_dir, config_name, resume=""):
    """Get the directory to save for corresponding `config`. 

    .. note:: If `saved_dir` in config is already exists and resume is `False`,
              it will remove `saved_dir`.
    
    Args:
        config (dict): Configuration dict.
        inner_dir (str): Directory between `config/` and configuration file.
        config_name (str): Configuration filename.
        resume (str): Path to checkpoint or False which means training from scratch (default: "").
    
    Returns:
        saved_dir (str): The directory to save.
        log_dir (str): The directory to save logs.
    """
    assert os.path.exists(config["saved_dir"])
    saved_dir = os.path.join(config["saved_dir"], inner_dir, config_name)
    if os.path.exists(saved_dir) and resume == "False":
        print("Delete existing {} for not resuming.".format(saved_dir))
        shutil.rmtree(saved_dir)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    log_dir = os.path.join(saved_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return saved_dir, log_dir


def get_storage_dir(config, inner_dir, config_name, resume=""):
    """Get the storage and checkpoint directory for corresponding `config`.

    .. note:: If `storage_dir` in config is already exists and resume is `False`,
              it will remove `storage_dir`.
    
    Args:
        config (dict): Configuration dict.
        inner_dir (str): Directory between `config/` and configuration file.
        config_name (str): Configuration filename.
        resume (str): Path to checkpoint or False which means training from scratch (default: "").
    
    Returns:
        storage_dir (str): Storage directory.
        ckpt_dir (str): Checkpoint directory.
    """
    assert os.path.exists(config["storage_dir"])
    storage_dir = os.path.join(config["storage_dir"], inner_dir, config_name)
    if os.path.exists(storage_dir) and resume == "False":
        print("Delete existing {} for not resuming.".format(storage_dir))
        shutil.rmtree(storage_dir)
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    ckpt_dir = os.path.join(storage_dir, "checkpoint")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    record_dir = os.path.join(storage_dir, "record")
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    return storage_dir, ckpt_dir, record_dir


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation.
            """
            pass

        return no_op


def get_logger(log_dir, log_name, resume, is_rank0=True):
    # Only log rank 0 in ddp training.
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

        # FileHandler
        if resume == "False":
            mode = "w+"
        else:
            mode = "a+"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
        start_time = time.asctime(time.localtime(time.time()))
        logger.info("Start at: {} at: {}".format(start_time, platform.node()))
    else:
        logger = NoOp()

    return logger


def set_seed(seed=None, deterministic=True, benchmark=False):
    """See https://pytorch.org/docs/stable/notes/randomness.html
    for detailed informations.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
