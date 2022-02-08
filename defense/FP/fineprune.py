import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
import copy
from tqdm import tqdm
# from config import get_arguments
from utils import args
import numpy as np
import torch.nn.functional as F

import sys
import pdb

sys.path.insert(0, "../..")
from utils.dataloader import get_dataloader
from utils.utils import progress_bar, save_checkpoint

# from networks.models import Normalizer, Denormalizer, NetC_MNIST ####
# from utils.resnet import ResNet18
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

def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()
    # losses = AverageMeter()
    # acc = AverageMeter()
    f = open(arg.log, "a")
    f.write("Training.\n")
    total_clean = 0
    total_clean_correct = 0
    train_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        if arg.classifier == 'preactresnet18':
            features, outputs = model(inputs)
        else:
            outputs = model(inputs)

        # # Label smoothing
        # hard_labels = to_one_hot(arg, labels)
        # ls_labels = 0.5 * hard_labels + 0.5 * ((torch.ones(hard_labels.shape) / arg.num_classes).to(arg.device))
        # loss = torch.mean(SoftCrossEntropy(outputs, ls_labels))

        # # Forget strategy
        # perm = torch.randint(0, inputs.shape[0], (int(inputs.shape[0] * 0.1),)).to(arg.device)
        # new_labels = torch.randint(0, arg.num_classes, (perm.shape[0],)).to(arg.device)
        # labels[perm] = new_labels

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # top1_acc = accuracy(outputs, labels)
        # losses.update(loss.item(), inputs.size(0))
        # acc.update(top1_acc, inputs.size(0))
        train_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (
        epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    scheduler.step()
    return train_loss / (i + 1), avg_acc_clean

# def eval(netC, test_dl, opt):
#     # print(" Eval:")
#     acc = 0.0
#     total_sample = 0
#     total_correct = 0
#
#     for index, (inputs, labels) in enumerate(test_dl):
#         inputs, labels = inputs.to(opt.device), labels.to(opt.device)
#         bs = inputs.shape[0]
#         total_sample += bs
#
#         # Evaluating clean or backdoor depending on the test_dl
#         preds = netC(inputs)
#         num_correct = torch.sum(torch.argmax(preds, 1) == labels)
#         total_correct += num_correct
#         acc = float(total_correct.item() * 100.0 / total_sample)
#
#         #### Evaluating backdoor
#         # grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
#         # grid_temps = torch.clamp(grid_temps, -1, 1)
#         # inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
#
#         # targets_bd = create_targets_bd(labels, opt)
#         # preds_bd = netC(inputs_bd)
#         # correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
#         # total_correct_bd += correct_bd
#         # acc_bd = total_correct_bd * 100.0 / total_sample
#         # progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
#     return acc

def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()
    # losses = AverageMeter()
    # acc = AverageMeter()
    f = open(arg.log, "a")
    f.write("Testing.\n")
    total_clean, total_clean_correct, total_robust_correct, test_loss = 0, 0, 0, 0
    total_clean_correct_t, total_clean_t = 0, 0
    for i, (inputs, labels, isCleans, gt_labels) in enumerate(testloader):
        inputs1, labels, isCleans, gt_labels = inputs.to(arg.device), labels.to(arg.device), isCleans.to(arg.device), gt_labels.to(arg.device)

        if arg.classifier == 'preactresnet18':
            features1, outputs1 = model(inputs1)
        else:
            outputs1 = model(inputs1)
        loss = criterion(outputs1, labels)
        # top1_acc = accuracy(outputs1, labels)
        # losses.update(loss.item(), inputs1.size(0))
        # acc.update(top1_acc, inputs1.size(0))
        test_loss += loss.item()
        ### 计算对于整体的效果 ###
        total_clean_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == labels[:])
        total_robust_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == gt_labels[:])
        total_clean += inputs1.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        avg_acc_robust = float(total_robust_correct.item() * 100.0 / total_clean)
        if word == 'bd':
            progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d) | Robust Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust, total_robust_correct, total_clean))
        if word == 'clean':
            # ### 计算对于target label的效果 ###
            # target_idx = np.where((gt_labels == arg.target_label).cpu().numpy())[0]
            # total_clean_correct_t += torch.sum(torch.argmax(outputs1[target_idx], dim=1) == arg.target_label)
            # total_clean_t += target_idx.shape[0]
            # avg_acc_clean_t = float(total_clean_correct_t.item() * 100.0 / total_clean_t)
            # progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d) | Target Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_clean_t, total_clean_correct_t, total_clean_t))
            progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean, avg_acc_robust


def main():
    # Prepare arguments
    global opt
    # opt = get_arguments()
    opt = args.get_args()

    #### load parameters into the model based on opt.dataset
    # if opt.dataset == "mnist":
    #     opt.input_height = 28
    #     opt.input_width = 28
    #     opt.input_channel = 1
    #    netC = NetC_MNIST().to(opt.device)
    #### elif opt.dataset == "CIFAR10" etc.
    # else:
    #    raise Exception("Invalid Dataset")
    netC = get_network(opt)  #### consistent with how train_attack.py creates model

    mode = opt.target_type
    # opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    # opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    # opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir") #### useless
    # ckpt_path = 'saved/checkpoint/checkpoint_' + opt.dataset + '.tar'
    optimizer = torch.optim.SGD(netC.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    if opt.checkpoint_load is not None:
        # checkpoint = torch.load(opt.checkpoint_load)
        ckpt_path = opt.checkpoint_load
        print("Continue training...")
        # model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = 0

    state_dict = torch.load(ckpt_path)  #### load parameters
    #### print(state_dict.keys())
    # exit(0)
    print("load C")
    netC.load_state_dict(state_dict["model"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    # print("load grid")
    # identity_grid = state_dict["identity_grid"].to(opt.device)
    # noise_grid = state_dict["noise_grid"].to(opt.device)
    # print(state_dict["best_clean_acc"], state_dict["best_bd_acc"])

    # Prepare dataloader and check initial acc_clean and acc_bd
    trainloader = get_dataloader(opt, True)
    testloader_clean, testloader_bd = get_dataloader_test(opt)  ####
    # test_loss, test_acc_cl, _ = test_epoch(opt, testloader_clean, netC, criterion, 0, 'clean')
    # test_loss, test_acc_bd, test_acc_robust = test_epoch(opt, testloader_bd, netC, criterion, 0, 'bd')
    # acc_clean = eval(netC, testloader_clean, opt)
    # acc_bd = eval(netC, testloader_bd, opt)
    # acc_robust = eval(netC, testloader_robust, opt)
    # print('Acc Clean: {:.3f} | Acc Bd: {:.3f} | Acc Robust: {:.3f}'.format(acc_clean, acc_bd, acc_robust))

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
    acc_clean = []
    acc_bd = []
    # opt.outfile = "{}_results.txt".format(opt.dataset)
    # with open(opt.outfile, "w") as outs:
    for index in range(int(pruning_mask.shape[0]*0.8)):
    # for index in range(101):
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

    test_loss, test_acc_cl, _ = test_epoch(opt, testloader_clean, net_pruned, criterion, 0, 'clean')
    test_loss, test_acc_bd, test_acc_robust = test_epoch(opt, testloader_bd, net_pruned, criterion, 0, 'bd')
        # acc_clean = eval(net_pruned, testloader_clean, opt)
        # acc_bd = eval(net_pruned, testloader_bd, opt)
        # acc_robust = eval(net_pruned, testloader_robust, opt)
        # print('Acc Clean: {:.3f} | Acc Bd: {:.3f} | Robust Bd: {:.3f}'.format(acc_clean, acc_bd, acc_robust))
        # clean, bd = eval(net_pruned, identity_grid, noise_grid, test_dl, opt)
        # outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))
    save_checkpoint(opt.checkpoint_save, index, net_pruned, optimizer, scheduler)
        # outs.write("%d %0.4f %0.4f\n" % (index, acc_clean, acc_bd))

    # ### Tuning ###
    # best_acc = -1000.0
    # criterion = nn.CrossEntropyLoss()
    # for epoch in tqdm(range(start_epoch, opt.epochs)):
    #     train_loss, train_acc = train_epoch(opt, trainloader, net_pruned, optimizer, scheduler, criterion, epoch)
    #     test_loss, test_acc_cl, _ = test_epoch(opt, testloader_clean, net_pruned, criterion, epoch, 'clean')
    #     test_loss, test_acc_bd, test_acc_robust = test_epoch(opt, testloader_bd, net_pruned, criterion, epoch, 'bd')
    #     # acc_clean = eval(net_pruned, testloader_clean, opt)
    #     # acc_bd = eval(net_pruned, testloader_bd, opt)
    #     # acc_robust = eval(net_pruned, testloader_robust, opt)
    #     # print('Acc Clean: {:.3f} | Acc Bd: {:.3f} | Acc Robust: {:.3f}'.format(acc_clean, acc_bd, acc_robust))
    #
    #     # if test_acc > best_acc:
    #     #     best_acc = test_acc
    #     #     save_checkpoint(arg.checkpoint_save, epoch, model, optimizer, scheduler)
    #
    #     if train_acc > best_acc:
    #         best_acc = train_acc
    #         save_checkpoint(opt.checkpoint_save, epoch, net_pruned, optimizer, scheduler)


if __name__ == "__main__":
    main()
