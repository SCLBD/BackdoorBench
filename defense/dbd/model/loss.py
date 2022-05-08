import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """Borrowed from https://github.com/wvangansbeke/Unsupervised-Classification.
    """

    def __init__(self, temperature, reduction="mean"):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert n == 2
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        if self.reduction == "mean":
            loss = -((mask * log_prob).sum(1) / mask.sum(1)).mean()
        elif self.reduction == "none":
            loss = -((mask * log_prob).sum(1) / mask.sum(1))
        else:
            raise ValueError("The reduction must be mean or none!")

        return loss


class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss.
    """

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class SCELoss(nn.Module):
    """Symmetric Cross Entropy.
    """

    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss

        return loss


class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u
