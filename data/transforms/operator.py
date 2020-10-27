import sys
import cv2
import torch
import random
import numpy as np
import sys
sys.path[0] = '/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet'
from utils.common import get_boxes, get_target, draw_image


# class EqRatioResize(object):
    
#     def __init__(self, cfg):
#         self.resize = cfg.INPUT.size


#     def __call__(self, img, target=None)


class CvColorJitter(object):
    """
    :param brightness, contrast and saturation of an image
    :return images after transforms
    """
    def __init__(self,
                 cfg=None,
                 brightness=32.0,
                 contrast=0.4,
                 staturation=10.0,
                 hue=(-0.5, 1.5)):

        self.brightness = (-brightness, brightness)
        self.contrast = (1 - contrast, 1 + contrast)
        self.staturation = (-staturation, staturation)
        self.hue = hue


        # self.brightness = (-cfg.INPUT.COLORJITTER.BRIGHTNESS,
        #                    cfg.INPUT.COLORJITTER.BRIGHTNESS)
        # self.contrast = (cfg.INPUT.COLORJITTER.CONTRAST,
        #                  cfg.INPUT.COLORJITTER.CONTRAST)
        # self.staturation = (-cfg.INPUT.COLORJITTER.SATURATION,
        #                     cfg.INPUT.COLORJITTER.SATURATION)
        # self.hue = cfg.INPUT.COLORJITTER.HUE

    def _distort(self, image, alpha=1.0, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def _convert(self, img):

        #convert brightness
        if random.randrange(2):
            self._distort(img,
                          beta=random.uniform(self.brightness[0],
                                              self.brightness[1]))

        #convert contrast
        if random.randrange(2):
            self._distort(img,
                          alpha=random.uniform(self.contrast[0],
                                               self.contrast[1]))

        # convert staturation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if random.randrange(2):
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.uniform(
                self.staturation[0], self.staturation[1])) % 180

        # convert hue
        if random.randrange(2):
            self._distort(img[:, :, 1],
                          alpha=random.uniform(self.hue[0], self.hue[1]))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        return img

    def __call__(self, img, target=None):
        """
        :param img:
        :param target:
        :return:
            cv image: Color jittered image.
        """

        return self._convert(img), target


class CvHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, cfg=None, p=1):
        self.p = p

    def __call__(self, img, target):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            # target[:, :4][:, 0:4:2] = img.shape[1] - target[:, :4][:, 0:4:2]

            af_x2 = img.shape[1] - target[:, :4][:, 0]
            af_x1 = img.shape[1] - target[:, :4][:, 2]
            target[:, :4][:, 0] = af_x1
            target[:, :4][:, 2] = af_x2
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CvBlurry(object):
    """
    make image Blurry, you can choose blur, GaussianBlur
    """
    def __init__(self, cfg=None, p = 0.3, blurtype = "GaussianBlur"):
        self.cfg = cfg
        self.p = p
        self.type = blurtype
    
    def __call__(self, img, target=None):
        if self.type == "GaussianBlur":
            kernels = [(3,3),(15,15)]
            lens = np.random.choice([i for i in range(len(kernels))])
            kernel = kernels[lens]
            img = cv2.GaussianBlur(img, kernel, 0)

        return img, target


class CvToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, cfg=None):
        pass

    def __call__(self, img, target=None):
        img = torch.from_numpy(img)
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '()'



class CvNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, cfg=None):
        # self.mean = cfg.INPUT.PIXEL_MEAN
        # self.std = cfg.INPUT.PIXEL_STD
        self.mean = [0.406, 0.456, 0.485]
        self.std = [0.225, 0.224, 0.229]

    def __call__(self, tensor, target=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor.astype(np.float32) / 255.0
        tensor -= self.mean
        tensor /= self.std
        tensor = tensor.transpose(2, 0, 1)
        return tensor, target


class CvResize(object):

    def __init__(self, cfg=None):
        self.resize = (960, 576)

    def __call__(self, img, target):

        h, w , channel = img.shape
        img = cv2.resize(img, self.resize)
        target[:, 0:4:2] = target[:, 0:4:2] * self.resize[0] / w
        target[:, 1:4:2] = target[:, 1:4:2] * self.resize[1] / h
    
        return img, target



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = eval(t)()(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# class CvEqResize(object):

#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.resize_shape = cfg.INPUT.SIZE
    
#     def __call__(self, img, target):


def eqratio_resize(img, resize_shape, target=None):
    """
    args:
        target: num_boxes x 4 (the format of boxes is xyxy)
        img: 3 x img_h x img_w
        resize_shape: (img_h, img_w)
        attention: all input augments is torch.Tensor
    return:
        img: 3 x img_h x img_w
        boxes: num_boxes x 4 (the format of boxes is xyxy)
    """
    _, img_h, img_w = img.shape
    img = img.unsqueeze(0)
    ex_w, ex_h = resize_shape
    src_ratio = float(img_w) / img_h
    ex_ratio = float(ex_w) / ex_h
    if (ex_ratio < src_ratio):
        resize_ratio = float(ex_w) / img_w
    else:
        resize_ratio = float(ex_h) / img_h
    resize_img_w = int(img_w * resize_ratio + 0.5)
    resize_img_h = int(img_h * resize_ratio + 0.5)
    img = F.interpolate(img, size=(resize_img_h, resize_img_w),
                        mode="nearest").squeeze(0)

    pad_w_left = (ex_w - resize_img_w) // 2
    pad_w_right = ex_w - resize_img_w - pad_w_left
    pad_h_top = (ex_h - resize_img_h) // 2
    pad_h_bottom = ex_h - resize_img_h - pad_h_top
    pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    img = F.pad(img, pad, "constant", value=0)
    if (target is not None):
        boxes = target.clone()
        boxes *= resize_ratio
        boxes[:, 0] += pad_w_left
        boxes[:, 1] += pad_h_top
        boxes[:, 2] += pad_w_left
        boxes[:, 3] += pad_h_top
        return img, boxes, pad, resize_ratio
    else:
        return img, _, pad, resize_ratio


if __name__ == "__main__":
    import cv2
    annotations_str = "/workspace/mnt/storage/dingrui/traffic-detection-data/Traffic_Fu_Xu/Annotations/panyu_2019-08-12-12-20-0526.xml"
    class_dic = ["person", 'non-motor', 'car']
    target = get_target(class_dic, annotations_str)
    img = cv2.imread("/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet/data/transforms/example.bmp")
    # draw_image(img, target)
    transfor = CvHorizontalFlip()
    # transfor = CvColorJitter()
    # transfor = CvBlurry(p = 1)
    # transfor = CvResize()
    # img, target = transfor(img, target)

    # trans = Compose(["CvColorJitter", "CvHorizontalFlip", "CvBlurry", "CvResize", "CvNormalize","CvToTensor"])
    # trans = Compose(["CvColorJitter"])
    img, target = transfor(img, target)

    draw_image(img, target)


    print("end")








    
