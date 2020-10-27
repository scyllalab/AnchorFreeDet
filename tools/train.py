

import os
import sys
import cv2
import torch
import argparse
import numpy as np
import torchcontrib
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import sys
sys.path[0] = '/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet'
from cfgs import cfg
from utils.logger import setup_logger
from utils.common import synchronize
from modeling.build_model import CenterNet
from data.datasets.dataset_example import DetectionsVOC
from data.transforms.operator import Compose






def preset(cfg, board_writer, local_rank, distributed):
    #data
    trans = Compose(["CvColorJitter", "CvHorizontalFlip", "CvBlurry", "CvResize", "CvNormalize","CvToTensor"])
    detectionvoc = DetectionsVOC(cfg, "trainV15B", trans)
    train_dataloader = torch.utils.data.DataLoader(detectionvoc, batch_size=cfg.SOLVER.BATCHSIZE, num_workers=1, shuffle=True, collate_fn=detectionvoc.collate_fn)

    #model
    model = CenterNet(cfg)
    torch.cuda.set_device(local_rank)
    model.cuda()

    #optimizer
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.005)

    synchronize()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    for j in range(cfg.SOLVER.MAX_EPOCH):
        model.train()
        for i, (imgs, targets) in enumerate(train_dataloader):
            out = model(imgs, targets)
            if(local_rank ==0):
                print(out)
            opt.zero_grad()
            out.backward()
            opt.step()
        opt.swap_swa_sgd()
        model.eval()
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.OUTDIR, "model_{}.pth".format(j)))
    


    







if __name__=="__main__":

    # parse args
    parser = argparse.ArgumentParser(description="AnchorFreeDet Training")
    parser.add_argument("--cfg", default="/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet/cfgs/example.yml", help="the path of config file")
    parser.add_argument("opts", help="Modify config options using the terminal-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.cfg != "":
        cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    
    """
    测试
    """
    # model = CenterNet(cfg)
    # trans = Compose(["CvColorJitter", "CvHorizontalFlip", "CvBlurry", "CvResize", "CvNormalize","CvToTensor"])
    
    # detectionvoc = DetectionsVOC(cfg, "val", trans)

    # train_dataloader = torch.utils.data.DataLoader(detectionvoc, batch_size=cfg.SOLVER.BATCHSIZE, num_workers=1, shuffle=False, collate_fn=detectionvoc.collate_fn)

    # for i, (imgs, targets) in enumerate(train_dataloader):
    #     out = model(imgs, targets)
    #     print(out)
    #     if i == 10:
    #         break




    # device seting
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    num_gpus = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()


    # poster and log
    output_dir = cfg.OUTPUT.OUTDIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("AnchorFreeDet", output_dir, 0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    with open(args.cfg, 'r') as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    board_writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))  # 数据存放在这个文件夹
    
    cudnn.benchmakr = True

    preset(cfg, board_writer, args.local_rank, args.distributed)