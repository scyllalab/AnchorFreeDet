import os
import cv2
import torch
import numpy as np
import sys
sys.path[0] = '/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet'
from torch.utils.data import Dataset
from utils.common import get_target, refine_label


class DetectionsVOC(Dataset):

    def __init__(self, cfg, txtname, transforms=None):
        self.cfg = cfg
        self.datadir = cfg.INPUT.DATADIR
        self.transforms = transforms
        self.txtname = txtname
        self.template_anno_str = os.path.join("%s", "Annotations", "%s.xml")
        self.template_image_str = os.path.join("%s", "ori_images", "%s.jpg")
        self.ids = list()

        for line in open(os.path.join(self.datadir, "ImageSets/Main", txtname+".txt")).readlines():
            self.ids.append((self.datadir, line.strip('\n')))
        # self.ids = self.ids[:200]
        
    def __getitem__(self, index):
        image_id = self.ids[index]
        img_path = self.template_image_str % image_id
        img = cv2.imread(img_path)
        target = get_target(self.cfg.MODEL.CLASSNAME, self.template_anno_str % image_id)
        img, target = self.transforms(img, target) 
        gt, categories_heatmaps = refine_label(self.cfg, target)

        return img_path, img, gt, categories_heatmaps

    def __len__(self):
        return len(self.ids)


    def collate_fn(self, batch):
        paths, imgs, gt, categories_heatmap = zip(*batch)
        imgs = torch.stack(imgs)
        categories_heatmaps = torch.stack(categories_heatmap)
        return imgs, (paths, gt, categories_heatmaps)



if __name__ == "__main__":

    print(sys.path)
    from cfgs import cfg
    from data.transforms.operator import Compose
    cfg.merge_from_file('./cfgs/example.yml')
    cfg.freeze()
    trans = Compose(["CvColorJitter", "CvHorizontalFlip", "CvBlurry", "CvResize", "CvNormalize","CvToTensor"])
    
    detectionvoc = DetectionsVOC(cfg, "val", trans)

    train_dataloader = torch.utils.data.DataLoader(detectionvoc, batch_size=cfg.SOLVER.BATCHSIZE, num_workers=1, shuffle=False, collate_fn=detectionvoc.collate_fn)

    for i, (paths, imgs, gt, categories_heatmaps) in enumerate(train_dataloader):
        import pdb
        pdb.set_trace()
        print(i)
        







