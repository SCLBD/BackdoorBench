import math
import torch.optim as optim
import torch
import numpy as np
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../')

# def set_args(args,module='sscl'):
#     if module == 'sscl':
#         args.batch_size = 512
#         args.learning_rate = 0.5
#         args.temp = 0.1
#         args.epochs = 200
#         args.num_workers = 16
#         args.method = 'SupCon' # choices = ['SupCon', 'SimCLR']
#         args.consine = True
        
#     elif module == 'mixed_ce':
#         args.batch_size = 512
#         args.learning_rate = 5
#         args.epochs = 10
#         args.num_workers = 16
#         args.consine = False

#     if args.batch_size > 256:
#         args.warm = True
#     if args.warm:
#         args.warmup_from = 0.01
#         args.warm_epochs = 10
#         if args.cosine:
#             eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
#             args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
#                     1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
#         else:
#             args.warmup_to = args.learning_rate
#     if args.debug:
#         args.epochs = 2
#     return args

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def set_optimizer(opt, model,lr=None):
    if lr == None:
        lr = opt.lr
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    return optimizer

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    print('==> Successfully saved!')
    del state

def accuracy(output, target, topk=(1,)): # output: (256,10); target: (256)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk) # 5
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # pred: (256,5)
        pred = pred.t() # (5,256)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (5,256)

        res = []

        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = torch.flatten(correct[:k]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
