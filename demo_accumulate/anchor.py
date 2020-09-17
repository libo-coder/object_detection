# coding=utf-8
"""
绘制锚框，画图展示如何绘制边界框和锚框
@author: libo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math


def draw_rectangle(currentAxis, bbox, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    """ 定义画矩形框的程序
    :param currentAxis: 坐标轴，通过 plt.gca() 获取
    :param bbox: 边界框，包含四个数值的 list，[x1, y1, x2, y2]
    :param edgecolor: 边框线条颜色
    :param facecolor: 填充颜色
    :param fill: 是否填充
    :param linestyle: 边框线型
    :return:
    """
    # patches.Rectangle 需要传入左上角坐标，矩形区域的宽度，高度等参数
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1, linewidth=1,
                             edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


def draw_anchor_box(center, length, scales, ratios, img_height, img_width):
    """ 绘制锚框
    :param center: 以 center 为中心，产生一系列锚框
    :param length: 指定一个基准的长度
    :param scales: 包含多种尺度比例的 list
    :param ratios: 包含多种长宽比的 list
    :param img_height: 图片的高度
    :param img_width: 图片的宽度，生成的锚框不能超出图片尺寸之外
    :return:
    """
    bboxs = []
    for scale in scales:
        for ratio in ratios:
            h = length * scale * math.sqrt(ratio)
            w = length * scale / math.sqrt(ratio)
            x1 = max(center[0] - w/2., 0.)
            y1 = max(center[1] - h/2., 0.)
            x2 = min(center[0] + w/2. - 1.0, img_width - 1.0)
            y2 = min(center[1] + h/2. - 1.0, img_height - 1.0)
            print(center[0], center[1], w, h)
            bboxs.append([x1, y1, x2, y2])

    for bbox in bboxs:
        draw_rectangle(currentAxis, bbox, edgecolor='b')


if __name__ == '__main__':
    plt.figure(figsize=(10, 10))

    img_path = './test_imgs/object_002.png'
    img = imread(img_path)
    plt.imshow(img)

    # 使用 xyxy 格式表示物体真实框
    bbox1 = [21, 26, 446, 156]
    bbox2 = [75, 193, 392, 356]

    currentAxis = plt.gca()

    draw_rectangle(currentAxis, bbox1, edgecolor='r')
    draw_rectangle(currentAxis, bbox2, edgecolor='r')

    img_height, img_width = img.shape[:2]
    draw_anchor_box([263., 275.], 100, [2.0], [0.5, 1.0, 2.0], img_height, img_width)

    ######################## 以下为添加文字说明和箭头###############################
    plt.text(220, 153, 'G1', color='red', fontsize=20)
    plt.arrow(233, 153, 30, 40, color='red', width=0.001, length_includes_head=True, head_width=5, head_length=10, shape='full')

    plt.savefig('./debug/object_002.png')
    # plt.show()
