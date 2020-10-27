from yacs.config import CfgNode as CN


_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = "0,1"
_C.MODEL.NUMCLASS = 3
_C.MODEL.CLASSNAME = [
    "preson",
    "non-motor",
    "car",
]
_C.PRETRAIN_PATH = "./checkpoints"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet"

_C.MODEL.FPN = CN()
_C.MODEL.FPN.NAME = "DarkFpn"

_C.INPUT = CN()
_C.INPUT.SIZE = [960, 576]
_C.INPUT.IMAGEMEAN = [0.406, 0.456, 0.485]
_C.INPUT.IMAGESTD =  [0.225, 0.224, 0.229]
_C.INPUT.DATASETNAME = "TrainDetection"
_C.INPUT.DATADIR = "./data"
_C.INPUT.JITOPTS = ["CvColorJitter", "CvHorizontalFlip", "CvNormalize", "CvToTensor"]


_C.OUTPUT = CN()
_C.OUTPUT.DOWNTIMES = 4
_C.OUTPUT.OUTDIR = "./outdir"

_C.SOLVER = CN()
_C.SOLVER.BATCHSIZE = 8
_C.SOLVER.GAUSSIAN_RADIUS = 0
_C.SOLVER.MAX_EPOCH = 200

