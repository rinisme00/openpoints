""" Cross Entropy w/ smoothing or soft targets

Borrowed from Ross Wightman (https://www.github.com/timm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS


@LOSS.register_module()
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, label_smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.confidence = 1. - self.smoothing
        self.register_buffer('weight', torch.tensor(weight) if weight is not None else None)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        if self.weight is not None:
            nll_loss = nll_loss * self.weight[target]
            # also scale smooth loss by the mean weight to keep balance
            smooth_loss = smooth_loss * self.weight.mean()

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


@LOSS.register_module()
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, **kwargs):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
