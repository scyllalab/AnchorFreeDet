import cv2
import torch
import numpy as np
import torch.distributed as dist
import xml.etree.ElementTree as ET



def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()







def eq_rescale_boxes(boxes, now_shape, origin_shape):

    boxes = boxes.clone()
    img_w, img_h = origin_shape
    ex_h, ex_w = now_shape
    src_ratio = float(img_w) / img_h
    ex_ratio = float(ex_w) / ex_h
    if (ex_ratio < src_ratio):
        resize_ratio = float(ex_w) / img_w
    else:
        resize_ratio = float(ex_h) / img_h

    resize_img_w = int(img_w * resize_ratio + 0.5)
    resize_img_h = int(img_h * resize_ratio + 0.5)
    pad_w_left = (ex_w - resize_img_w) // 2
    pad_w_right = ex_w - resize_img_w - pad_w_left
    pad_h_top = (ex_h - resize_img_h) // 2
    pad_h_bottom = ex_h - resize_img_h - pad_h_top
    pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)

    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[2]
    boxes[:, 2] -= pad[0]
    boxes[:, 3] -= pad[2]

    boxes[:, 0] /= resize_ratio
    boxes[:, 1] /= resize_ratio
    boxes[:, 2] /= resize_ratio
    boxes[:, 3] /= resize_ratio
    return boxes

def get_boxes(json_file, num_classes=None):
    '''
    Considering the num classes of json file are more than the numbers you want to use, num_classes is used to remove the extra annotations.
    If you want to use all num classes, num_classes keep default is recommended.
    '''
    box_valid = []
    with open(json_file, 'r') as fr:
        json_data = json.load(fr)
        boxes = json_data['boxes']['boxes']
        target = json_data['labels']
        ids = json_data['ids']
        box_level = json_data['boxes']['box_level']
        if num_classes is not None and isinstance(num_classes, int):
            for i, gt_label in enumerate(target):
                if gt_label < num_classes:
                    box_valid.append([gt_label, *boxes[i]])
        else:
            for i, gt_label in enumerate(target):
                box_valid.append([gt_label, *boxes[i]])
        boxes = np.array(box_valid).astype(np.float32).reshape(-1, 5)
        return boxes, ids, box_level



def get_target(class_dic, anno_path):
    try:
        tree = ET.parse(anno_path)
    except:
        raise SystemExit("the annotation not found.")
    else:
        res = np.empty((0, 5), dtype=np.float32)
        for obj in tree.iter('object'):
            name = obj.find('name')
            if name is not None:
                name = name.text.lower().strip()
            else:
                continue
            if name not in class_dic:
                continue
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = list()
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            label_id = class_dic.index(name)
            bndbox.append(label_id)
            if len(bndbox)!=0:
                res = np.vstack((res,bndbox))
            else:
                res = np.vstack((res, [0, 0, 0, 0, -1]))
        return res

def draw_image(img, boxes):
    '''
    img: npdarrry
    boxes: nx5 ----> x1, y1, x2, y2, cls
    '''

    COLOR=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255]]
    if boxes is None:
        return
    
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1,y1),(x2,y2),COLOR[int(box[-1])],1)
        cv2.putText(img, str(int(box[-1])), (int(x1+6),y1), cv2.FONT_HERSHEY_SIMPLEX ,  0.5, COLOR[int(box[-1])], 1, cv2.LINE_AA)
    cv2.imwrite("./temp_0.jpg",img)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(dx, dy, sigma = 1):

    data = -(dx * dx + dy * dy) / (2 * sigma * sigma)

    value = np.exp(data)

    return value


def draw_gaussian(heatmap, center, radius, sigma= 1):

    width, height = heatmap.shape[:2]
    cx = center[0]
    cy = center[1]

    if radius < 1:
        heatmap[cx][cy] = 1.0
    else: 
        left = max(0, int(cx - radius))
        right = min(int(cx + radius), width)
        top = max(0, cy - radius)
        bottom = min(int(cy + radius), height)
        for i in range(left, right):
            for j in range(top, bottom):
                heatmap[i][j] = max(heatmap[i][j], gaussian2D((i - cx)/ radius, (j - cy)/ radius))







# def gaussian2D(shape, sigma=1):
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m+1,-n:n+1]

#     h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     return h

# def draw_gaussian(heatmap, center, radius, k=1, delte=6):
#     diameter = 2 * radius + 1
#     gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

#     x, y = center

#     height, width = heatmap.shape[0:2]
    
#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)

#     masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)





def  refine_label(cfg,  target):

    if target is None:
        return 
    uptimes = cfg.OUTPUT.DOWNTIMES
    categories = cfg.MODEL.NUMCLASS
    radius = cfg.SOLVER.GAUSSIAN_RADIUS
    #输出map大小, 宽高
    output_size = [cfg.INPUT.SIZE[0] / uptimes, cfg.INPUT.SIZE[1] / uptimes]
    categories_heatmaps = np.zeros((categories, int(output_size[0]), int(output_size[1])), dtype=np.float32)
    if target.shape[0] == 0:
        categories_heatmaps = torch.from_numpy(categories_heatmaps)
        return target, categories_heatmaps

    #转化为百分比
    target[:, 0:4:2] = target[:, 0:4:2] / cfg.INPUT.SIZE[0]
    target[:, 1:4:2] = target[:, 1:4:2] / cfg.INPUT.SIZE[1]
    cx = (target[:, 2] + target[:, 0]) / 2
    cy = (target[:, 3] + target[:, 1]) / 2
    gt_w = torch.from_numpy(target[:, 2] - target[:, 0]).unsqueeze(1) 
    gt_h = torch.from_numpy(target[:, 3] -  target[:, 1]).unsqueeze(1)
    mili = torch.cat((gt_w / 2 * output_size[0], gt_h / 2 * output_size[1]),1)
    try:
        _, radius = torch.min(mili, 1)
    except RuntimeError:
        print("nene")
        return
    
    
    pos_cx = cx * output_size[0]
    pos_cy = cy * output_size[1]
    # gt_offset_cx = torch.from_numpy(pos_cx - np.floor(pos_cx)).unsqueeze(1)
    # gt_offset_cy = torch.from_numpy(pos_cy - np.floor(pos_cy)).unsqueeze(1)

    for cls, int_cx, int_cy, radius_i in zip(*(target[:, -1], np.floor(pos_cx), np.floor(pos_cy), radius)):
        draw_gaussian(categories_heatmaps[int(cls)], [int(int_cx), int(int_cy)], radius_i)
    
    categories_heatmaps = torch.from_numpy(categories_heatmaps)

    pos_cx = torch.from_numpy(pos_cx).unsqueeze(1)
    pos_cy = torch.from_numpy(pos_cy).unsqueeze(1)

    return torch.cat((pos_cx, pos_cy, gt_w, gt_h), 1).float(), categories_heatmaps


    
