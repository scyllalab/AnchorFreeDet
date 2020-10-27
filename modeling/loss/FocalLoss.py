import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        index = target > 0.8


        pos_pre = pred[index]
        pos_gt = target[index]

        loss_pos = self.alpha*torch.pow(1-pos_pre, self.gamma)*pos_pre.clamp(min=0.0001, max=1.0).log()
        

        neg_pre = pred[~index]
        neg_gt = target[~index]

        loss_neg = (1-self.alpha)*torch.pow(neg_pre, self.gamma)*(1-neg_pre).clamp(min=0.0001, max=1.0).log()

        # focal_loss = loss_pos

        if self.size_average:
            focal_loss = -loss_pos.mean() - loss_neg.mean()
        else:
            focal_loss = loss_pos.sum()
        return focal_loss



        




        pred = torch.cat((1-pred, pred), dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        
        class_mask.scatter_(1, target.view(-1, 1).long(), 1)

