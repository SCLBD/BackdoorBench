import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import logging

from utils.aggregate_block.model_trainer_generate import partially_load_state_dict

from .loss import SimCLRLoss, SCELoss, MixMatchLoss
from .network import densenet_face, resnet_cifar, resnet_imagenet, preact_dbd, vgg_dbd, densenet_dbd, mobilenet_dbd, efficientnet_dbd


def get_network(network_config):
    if "resnet18_cifar" in network_config:
        model = resnet_cifar.resnet18(**network_config["resnet18_cifar"])
    elif "resnet18_imagenet" in network_config:
        model = resnet_imagenet.resnet18(**network_config["resnet18_imagenet"])
    elif "densenet121_face" in network_config:
        model = densenet_face.densenet121(**network_config["densenet121_face"])
    else:
        raise NotImplementedError("Network {} is not supported.".format(network_config))

    return model

def get_network_dbd(args):
    model = args.model
    if model == "preactresnet18":
        model = preact_dbd.PreActResNet18(args.num_classes)
    elif model == "vgg19":
        model = vgg_dbd.vgg19(num_classes = args.num_classes)
    elif model == "vgg19_bn":
        model = vgg_dbd.vgg19_bn(num_classes = args.num_classes)
    # elif "densenet121_face" in network_config:
    #     model = densenet_face.densenet121(**network_config["densenet121_face"])
    elif model == 'densenet161':
        model = densenet_dbd.densenet161(num_classes= args.num_classes)
    elif model == 'mobilenet_v3_large':
        model = mobilenet_dbd.mobilenet_v3_large(num_classes= args.num_classes)
    elif model == 'efficientnet_b3':
        model = efficientnet_dbd.efficientnet_b3(num_classes= args.num_classes)
    elif model == 'convnext_tiny':
        try:
            from torchvision.models import convnext_tiny
            from .network import conv_dbd 
            net_from_imagenet = convnext_tiny(pretrained=True, ) 
            model = conv_dbd.convnext_tiny(num_classes=args.num_classes)
            partially_load_state_dict(model, net_from_imagenet.state_dict())
        except:
            from torchvision.models import convnext_tiny
            from .network import conv_new_dbd 
            net_from_imagenet = convnext_tiny(pretrained=True, ) 
            model = conv_new_dbd.convnext_tiny(num_classes=args.num_classes)
            partially_load_state_dict(model, net_from_imagenet.state_dict())
    elif model == 'vit_b_16':
        from torchvision.models import vit_b_16
        from .network import vit_dbd 
        net_from_imagenet = vit_b_16(pretrained=True, ) 
        net = vit_dbd.vit_b_16()
        partially_load_state_dict(net, net_from_imagenet.state_dict())
        from torchvision.transforms import Resize
        model = torch.nn.Sequential(
            Resize((224, 224)),
            net,
        )
        model.feature_dim = net.feature_dim
    else:
        raise NotImplementedError("Network {} is not supported.".format(model))

    return model



def get_criterion(criterion_config):
    if "cross_entropy" in criterion_config:
        criterion = nn.CrossEntropyLoss(**criterion_config["cross_entropy"])
    elif "simclr" in criterion_config:
        criterion = SimCLRLoss(**criterion_config["simclr"])
    elif "sce" in criterion_config:
        criterion = SCELoss(**criterion_config["sce"])
    elif "mixmatch" in criterion_config:
        criterion = MixMatchLoss(**criterion_config["mixmatch"])
    else:
        raise NotImplementedError(
            "Criterion {} is not supported.".format(criterion_config)
        )

    return criterion


def get_optimizer(model, optimizer_config):
    if "Adam" in optimizer_config:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config["Adam"])
    elif "SGD" in optimizer_config:
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config["SGD"])
    else:
        raise NotImplementedError(
            "Optimizer {} is not supported.".format(optimizer_config)
        )

    return optimizer


def get_scheduler(optimizer, lr_scheduler_config):
    if lr_scheduler_config is None:
        scheduler = None
    elif "multi_step" in lr_scheduler_config:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, **lr_scheduler_config["multi_step"]
        )
    elif "cosine_annealing" in lr_scheduler_config:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, **lr_scheduler_config["cosine_annealing"]
        )
    else:
        raise NotImplementedError(
            "Learning rate scheduler {} is not supported.".format(lr_scheduler_config)
        )

    return scheduler


def load_state(
    model, resume, ckpt_dir, device,logger, optimizer=None, scheduler=None, is_best=False
):
    """Load training state from checkpoint.

    Args:
        model (torch.nn.Module): Model to resume.
        resume (string): Checkpoint name (empty string means the latest checkpoint)
            or False (means training from scratch).
        ckpt_dir (string): Checkpoint directory.
        device : GPU or CPU.
        ###logger (logging.logger): The logger.
        optimizer (torch.optim.Optimizer): Optimizer to resume (default: None).
        scheduler (torch.optim._LRScheduler): Learning rate scheduler to
            resume (default: None).
        is_best (boolean, optional): Set True to load checkpoint
            with `best_acc` (default: False).

    Returns:
        resumed_epoch: The epoch to resume (0 means training from scratch.)
        best_acc: The best test accuracy in the training.
        best_epoch: The epoch getting the `best_acc`.
    """
    if resume == "False":
        logging.warning("Training from scratch.")
        resumed_epoch = 0
        if is_best:
            best_acc = 0
            best_epoch = 0
            return resumed_epoch, best_acc, best_epoch
        else:
            return resumed_epoch
    else:
        # Load checkpoint.
        # if resume == "":
        #     ckpt_path = os.path.join(ckpt_dir, "latest_model.pt")
        # else:
        #     ckpt_path = os.path.join(ckpt_dir, resume)
        ckpt_path = ckpt_dir
        ckpt = torch.load(ckpt_path, map_location=device)
        # logger.info("Load training state from the checkpoint {}:".format(ckpt_path))
        # logger.info("Epoch: {}, result: {}".format(ckpt["epoch"], ckpt["result"]))
        if "parallel" in str(type(model)):
            # DataParallel or DistributedParallel.
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Remove "module." in `model_state_dict` if saved
            # from DDP wrapped model in the single GPU training.
            model_state_dict = OrderedDict()
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                    model_state_dict[k] = v
                else:
                    model_state_dict[k] = v
            model.load_state_dict(model_state_dict)
        resumed_epoch = ckpt["epoch"]
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if is_best:
            best_acc = ckpt["best_acc"]
            best_epoch = ckpt["best_epoch"]
            return resumed_epoch, best_acc, best_epoch
        else:
            return resumed_epoch


def get_saved_epoch(
    num_epochs, num_stage_epochs=100, min_interval=20, max_interval=100
):
    if num_epochs >= num_stage_epochs:
        early = set(range(min_interval, num_stage_epochs, min_interval))
        mid = set(range(num_stage_epochs, num_epochs - num_stage_epochs, max_interval))
        later = set(
            range(
                num_epochs - num_stage_epochs, num_epochs + min_interval, min_interval
            )
        )
        if num_epochs == num_stage_epochs:
            later.remove(0)
        saved_epoch = early.union(mid).union(later)
    else:
        raise ValueError(
            "The num_epochs: {} must be equal or greater than num_stage_epochs: {}".format(
                num_epochs, num_stage_epochs
            )
        )

    return saved_epoch
