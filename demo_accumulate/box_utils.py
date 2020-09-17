# coding=utf-8
"""
目标检测中常用的一些代码
@author: libo
"""
import numpy as np
import matplotlib.pyplot as plt

def box_iou_xyxy(box1, box2):
    """ 计算 IoU，矩形框的坐标形式为 xyxy
    :param box1: 左上角坐标
    :param box2: 右下角坐标
    :return: iou
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = box1[0], box1[1], box1[2], box1[3]
    s1 = (ymax_1 - ymin_1 + 1.) * (xmax_1 - xmin_1 + 1.)    # 计算 box1 的面积

    xmin_2, ymin_2, xmax_2, ymax_2 = box2[0], box2[1], box2[2], box2[3]
    s2 = (ymax_2 - ymin_2 + 1.) * (xmax_2 - xmin_2 + 1.)    # 计算 box2 的面积

    # 计算相交矩阵的坐标
    xmin = np.maximum(xmin_1, xmin_2)
    ymin = np.maximum(ymin_1, ymin_2)
    xmax = np.minimum(xmax_1, xmax_2)
    ymax = np.minimum(ymax_1, ymax_2)

    # 计算相交矩形的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w

    # 计算相并的面积
    union = s1 + s2 - intersection

    # 计算交并比
    iou = intersection / union
    return iou


def box_iou_xywh(box1, box2):
    xmin_1, ymin_1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    xmax_1, ymax_1 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    s1 = box1[2] * box1[3]

    xmin_2, ymin_2 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    xmax_2, ymax_2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(xmin_1, xmin_2)
    ymin = np.maximum(ymin_1, ymin_2)
    xmax = np.minimum(xmax_1, ymax_1)
    ymax = np.minimum(xmax_2, ymax_2)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou

if __name__ == '__main__':
    bbox1 = [100., 100., 200., 200.]
    bbox2 = [120., 120., 220., 220.]
    iou = box_iou_xyxy(bbox1, bbox2)
    print('IoU is {}'.format(iou))


