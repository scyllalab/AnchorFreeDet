import torch
import torch.nn as nn


class LLoss(nn.Module):

    def __init__(self, size_average=True):
        super(LLoss, self).__init__()
        self.size_average = size_average

    def forward(self, pre, target):
        
        pred = pre.contiguous().view(-1, 1)
        target = target.contiguous().view(-1, 1)
        

        gt_loss = torch.abs(pred - target).sum()

        if self.size_average:
            gt_loss = gt_loss.mean()

        return gt_loss