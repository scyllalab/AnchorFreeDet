import torch
import torch.nn as nn
from modeling.loss import FocalLoss, LLoss

class CenterLoss(nn.Module):

    def __init__(self, cfg):
        super(CenterLoss, self).__init__()
        self.cfg = cfg
        self.FLoss = FocalLoss()
        self.LLoss = LLoss()

    def forward(self, out, target):

        pre_heatmaps, pre_offsets, pre_whs = out
        paths, gts, categories_heatmaps = target
        batchsize = len(paths)
        loss = 0
        for i, (path, gt, heatmap) in enumerate(zip(*(paths, gts, categories_heatmaps))):
            if gt.shape[0] ==0:
                continue
            pre_offset = pre_offsets[i].permute(2,1,0)
            pre_wh = pre_whs[i].permute(2,1,0)
            pos_cxcy = gt[:, :2] 
            try:   
                int_cxcy = pos_cxcy.long() # 9x2
            except AttributeError:
                print("Error!")
            gt_offset = pos_cxcy - int_cxcy
            gt_wh = gt[:, 2:]
            pre_offset = pre_offset[int_cxcy[:, 0], int_cxcy[:, 1], :]
            pre_wh = pre_wh[int_cxcy[:, 0], int_cxcy[:, 1], :]
            

            focal_loss = self.FLoss(pre_heatmaps[i],  heatmap)
            offset_loss = self.LLoss(pre_offset, gt_offset)
            wh_loss = self.LLoss(pre_wh, gt_wh)

            loss += focal_loss + offset_loss + wh_loss
        loss /= batchsize
        return loss
        


        



