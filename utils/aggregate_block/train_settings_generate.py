# This script contains function to set the training criterion, optimizer and schedule.

import sys, logging
sys.path.append('../../')
import torch
import torch.nn as nn

class flooding(torch.nn.Module):
    # idea: module that can add flooding formula to the loss function
    '''The additional flooding trick on loss'''
    def __init__(self, inner_criterion, flooding_scalar = 0.5):
        super(flooding, self).__init__()
        self.inner_criterion = inner_criterion
        self.flooding_scalar = float(flooding_scalar)
    def forward(self, output, target):
        return (self.inner_criterion(output, target) - self.flooding_scalar).abs() + self.flooding_scalar

def argparser_criterion(args):
    '''
    # idea: generate the criterion, default is CrossEntropyLoss
    '''
    criterion = nn.CrossEntropyLoss()
    if ('flooding_scalar' in args.__dict__): # use the flooding formulation warpper
        criterion = flooding(
            criterion,
            flooding_scalar=float(
                            args.flooding_scalar
                        )
        )
    return criterion

def argparser_opt_scheduler(model, args):
    # idea: given model and args, return the optimizer and scheduler you choose to use

    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=args.sgd_momentum,  # 0.9
                                    weight_decay=args.wd,  # 5e-4
                                    )
    elif args.client_optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            rho = args.rho, #0.95,
            eps = args.eps, #1e-07,
        )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     betas=args.adam_betas,
                                     weight_decay=args.wd,
                                     amsgrad=True)

    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.min_lr,
                                                      max_lr=args.lr,
                                                      step_size_up= args.step_size_up,
                                                      step_size_down= args.step_size_down,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.steplr_stepsize,  # 1
                                                    gamma=args.steplr_gamma)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 if (("cos_t_max" not in args.__dict__) or args.cos_t_max is None) else args.cos_t_max)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplr_milestones ,args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                'factor':args.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in args.__dict__ else {})
        )
    else:
        scheduler = None

    return optimizer, scheduler
