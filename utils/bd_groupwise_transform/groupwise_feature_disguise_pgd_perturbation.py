
'''
from
@article{saha2019hidden,
  title={Hidden Trigger Backdoor Attacks},
  author={Saha, Aniruddha and Subramanya, Akshayvarun and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:1910.00033},
  year={2019}
}

code source : https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
'''
from typing import Optional

import torch, sys
sys.path.append('../../')

from utils.hook_forward_lastHiddenLayerActivationExtractor import lastHiddenActivationExtractor


# assistant function for pgd lr update, you can personalize this function.
def adjust_learning_rate(lr, iter):
    """Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
    lr = lr * (0.5 ** (iter // 1000))
    return lr

# this function is used to groupwise disguise patched source data as target data in pixel.
# but still keep them close in the feature space
def groupwise_feature_disguise_pgd_perturbation(
        patched_source_img_batch : torch.Tensor, # default batchsize = 100
        target_img_batch : torch.Tensor, # default batchsize = 100
        model : torch.nn.Module, # regular model, NO NEED to add extractor!
        device : torch.device,
        img_eps1 : float, # assume pixel value in [0,1]
        pgd_init_lr : float, # default = 0.01
        pgd_max_iter : int, # maximum iteration for pgd, default = 5000
) -> Optional[torch.Tensor]: # will return the perturbed patched_source_img_batch

    assert len(patched_source_img_batch) == len(target_img_batch) # must same, since we need one-to-one pairing

    model = model.to(device)
    model.eval()
    model_with_extractor = lastHiddenActivationExtractor(model)

    patched_source_img_batch = patched_source_img_batch.to(device)
    target_img_batch = target_img_batch.to(device)

    pert = torch.nn.Parameter(torch.zeros_like(target_img_batch, requires_grad=True).to(device))

    feat1, _ = model_with_extractor.generateLastLinearInput(patched_source_img_batch, device)
    feat1 = feat1.detach().clone()

    for j in range(pgd_max_iter):

        lr1 = adjust_learning_rate(pgd_init_lr, j)

        feat2, _ = model_with_extractor.generateLastLinearInput(target_img_batch + pert, device)

        # FIND CLOSEST PAIR WITHOUT REPLACEMENT
        feat11 = feat1.clone()
        # save the original copy of feat1
        dist = torch.cdist(feat1, feat2)
        for _ in range(feat2.shape[0]):
            dist_min_index = torch.nonzero(dist == torch.min(dist))#(dist == torch.min(dist)).nonzero().squeeze()
            # the output should be torch.tensor with (n,2) shape
            feat1[dist_min_index[0][1]] = feat11[dist_min_index[0][0]] #rearrange the vector in feat1 so that feat1 and feat2 have pairs match in index
            dist[dist_min_index[0][0], dist_min_index[0][1]] = 1e5

        loss1 = ((feat1 - feat2) ** 2).sum(dim=1)
        loss = loss1.sum()

        loss.backward()

        # PGD update the pert for each img

        pert = pert - lr1 * pert.grad
        pert = torch.clamp(pert, -img_eps1, img_eps1).detach_()

        pert = pert + target_img_batch # here pert changes ! actually the perturbed img batch !

        pert = pert.clamp(0, 1)

        # exit for function
        if loss1.max().item() < 10 or j == (pgd_max_iter - 1):
            return pert

        pert = pert - target_img_batch
        pert.requires_grad = True
        if pert.grad is not None:
            pert.grad.zero_()

    # if reach here, it means both ending condition fails for PGD.
    return None
