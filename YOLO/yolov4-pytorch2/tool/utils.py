import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool.torch_utils import get_region_boxes


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    # print('iou box1:', box1)
    # print('iou box2:', box2)
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    print(boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        print(box)
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def post_processing(img, conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9       # 3 个 yolo 层共 9 种锚点
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]     # 每个 yolo 层相对输入图像尺寸的减少倍数分别是 8，16，32
    # anchor_step = len(anchors) // num_anchors

    box_array = output[0]       # [batch, num, 1, 4]
    confs = output[1]       # [batch, num, num_classes]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    box_array = box_array[:, :, 0]      # [batch, num, 4]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh)
        
        bboxes = []
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]

            for j in range(l_box_array.shape[0]):
                bboxes.append([l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3],
                               l_max_conf[j], l_max_conf[j], l_max_id[j]])
        
        bboxes_batch.append(bboxes)

    t3 = time.time()

    print('-----------------------------------')
    print('       max and argmax : %f' % (t2 - t1))
    print('                  nms : %f' % (t3 - t2))
    print('Post processing total : %f' % (t3 - t1))
    print('-----------------------------------')
    
    return bboxes_batch


def post_processing2(img, conf_thresh, nms_thresh, use_cuda=1):
    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9       # 3 个 yolo 层共 9 种锚点
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]     # 每个 yolo 层相对输入图像尺寸的减少倍数分别是 8，16，32
    anchor_step = len(anchors) // num_anchors

    boxes = []
    for i in range(3):
        masked_anchors = []
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step: (m+1) * anchor_step]
        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]

        """
        模型 forward 后输出结果存在 list_boxes 中，因为有 3 个 yolo 输出层，所以 list_boxes 又分为 3 个子列表
        其中 list_boxes[0] 中存放的是第一个 yolo 层输出，其特征图大小对于原图缩放尺寸为 8，即 strides[0], 
        对于 608x608 图像来说，该层的 featuremap 尺寸为 608/8=76；
        则该层的yolo输出数据维度为[batch, (classnum+4+1)*num_anchors, feature_h, feature_w] , 
        对于80类的coco来说，测试图像为1，每个yolo层每个特征图像点有3个锚点，该yolo层输出是[1,255,76,76]；
        对应锚点大小为[1.5,2.0,2.375,4.5,5.0,3.5]; (这6个数分别是3个锚点的w和h,按照w1,h1,w2,h2,w3,h3排列)；
        
        同理第二个 yolo 层检测结果维度为[1,255,38,38],对应锚点大小为：[2.25,4.6875,4.75,3.4375,4.5,9.125],输出为 [1,255,38,38]；
        
        第三个yolo层检测维度为[1,255,19,19]，对应锚点大小为：[4.4375,3.4375,6.0,7.59375,14.34375,12.53125]，输出为 [1,255,19,19]；
        
        get_region_boxes1(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False) 
        这个函数对 forward 后的 output 做解析并做 nms 操作；

        """
        boxes.append(get_region_boxes(list_boxes[i].data.numpy(), 0.6, 80, masked_anchors, len(anchor_masks[i])))

    if img.shape[0] > 1:
        bboxs_for_imgs = [boxes[0][index] + boxes[1][index] + boxes[2][index] for index in range(img.shape[0])]
        boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]        # 分别对每一张图像做nms
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        boxes = nms(boxes, nms_thresh)

    # 返回nms后的boxes
    return boxes
