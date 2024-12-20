import torch.nn as nn

from . import functional as F

__all__ = ['KLLoss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        print("In loss: KLLoss()nn.Module")
        return F.kl_loss(x, y)
