import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

from utils import args
from utils.dataloader import get_dataloader
from utils.utils import progress_bar, save_checkpoint
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_test


def create_targets_bd(labels, opt):
    if opt.target_type == "all2one":
        bd_targets = torch.ones_like(labels) * opt.target_label
    elif opt.target_type == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in labels])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.target_type))
    return bd_targets.to(opt.device)


def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()

    total_clean, total_clean_correct, test_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

        if word == 'bd':
            progress_bar(i, len(testloader), 'Test %s ASR: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        if word == 'clean':
            progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean


def main():
    global opt
    opt = args.get_args()

    # Prepare model, optimizer, scheduler
    netC = get_network(opt)
    optimizer = torch.optim.SGD(netC.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    ckpt_path = opt.checkpoint_load
    print("Start finepruning model...")
    state_dict = torch.load(ckpt_path)
    netC.load_state_dict(state_dict["model"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()

    # Prepare dataloader and check initial acc_clean and acc_bd
    trainloader = get_dataloader(opt, True)
    testloader_clean, testloader_bd = get_dataloader_test(opt)

    for name, module in netC._modules.items():
        print(name)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    if opt.classifier == 'preactresnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)
    if opt.classifier == 'vgg19_bn':
        hook = netC.features.register_forward_hook(forward_hook)
    if opt.classifier == 'resnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(trainloader))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    for index in range(int(pruning_mask.shape[0])):
        net_pruned = copy.deepcopy(netC)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
        print("Pruned {} filters".format(num_pruned))

        if opt.classifier == 'preactresnet18':
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
        if opt.classifier == 'vgg19_bn':
            net_pruned.features[49] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.features[50] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.classifier[0] = nn.Linear(pruning_mask.shape[0] - num_pruned, 4096)
        if opt.classifier == 'resnet18':
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.fc = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

        # Re-assigning weight to the pruned net
        for name, module in net_pruned._modules.items():
            if opt.classifier == 'preactresnet18':
                if "layer4" == name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            if opt.classifier == 'vgg19_bn':
                if "features" == name:
                    module[49].weight.data = netC.features[49].weight.data[pruning_mask]
                    module[49].ind = pruning_mask
                    module[50].weight.data = netC.features[50].weight.data[pruning_mask]
                    module[50].ind = pruning_mask
                elif "classifier" == name:
                    module[0].weight.data = netC.classifier[0].weight.data[:, pruning_mask]
                    module[0].bias.data = netC.classifier[0].bias.data
                else:
                    continue
            if opt.classifier == 'resnet18':
                if "layer4" == name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "fc" == name:
                    module.weight.data = netC.fc.weight.data[:, pruning_mask]
                    module.bias.data = netC.fc.bias.data
                else:
                    continue
        net_pruned.to(opt.device)

    test_loss, test_acc_cl = test_epoch(opt, testloader_clean, net_pruned, criterion, 0, 'clean')
    test_loss, test_acc_bd = test_epoch(opt, testloader_bd, net_pruned, criterion, 0, 'bd')

    save_checkpoint(opt.checkpoint_save, index, net_pruned, optimizer, scheduler)


if __name__ == "__main__":
    main()
