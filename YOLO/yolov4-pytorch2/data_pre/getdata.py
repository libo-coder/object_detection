# coding=utf-8
"""
目的是：在主目录下生成两个文件夹 train.txt 和 val.txt
"""
import xml.etree.ElementTree as ET
import pickle
import os
import sys
from os import listdir, getcwd
from os.path import join


# 这里是将左上角和右下角的坐标转化为中心点和宽高
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(wd + '/Annotations/%s.xml' % image_id)
    out_file = open(wd + '/labels/%s.txt' % image_id, 'w')
    tree=ET.parse(in_file)      # 导入xml数据
    root = tree.getroot()       # 得到跟节点
    size = root.find('size')    # 找到根节点下面的size节点
    w = int(size.find('width').text)    # 得到图片的尺寸
    h = int(size.find('height').text)

    for obj in root.iter('object'):     # 对根节点下面的'object’节点进行遍历
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



sets = ['train', 'val']
classes = ['bar']    # 这里是你要处理的数据的类别总数


if __name__=='__main__':
    ############################## raw ##################################
    # wd = getcwd()   # 获取当前文件的路径
    # wd = wd.replace('\\', '/')
    # for image_set in sets:  # image_set是train或者val
    #     if not os.path.exists(wd + '/labels/'):     # 创建一个label文件夹来存放图片对应的类别和坐标
    #         os.makedirs(wd + '/labels/')
    #     image_ids = open(wd +'/ImageSets/Main/%s.txt' % image_set).read().strip().split()
    #     list_file = open('%s.txt' % image_set, 'w')
    #     for image_id in image_ids:
    #         list_file.write(wd + '/JPEGImages/%s.jpg\n' % image_id)
    #         convert_annotation(image_id)
    #     list_file.close()
    ######################################################################
    wd = '../VOCdevkit_lb/VOC0807'

    for image_set in sets:  # image_set是train或者val
        if not os.path.exists('../VOCdevkit_lb/VOC0807/labels/'):  # 创建一个label文件夹来存放图片对应的类别和坐标
            os.makedirs('../VOCdevkit_lb/VOC0807//labels/')
        image_ids = open('../VOCdevkit_lb/VOC0807/ImageSets/Main/%s.txt' % image_set).read().strip().split()
        list_file = open('%s.txt' % image_set, 'w')
        for image_id in image_ids:
            list_file.write('../VOCdevkit_lb/VOC0807/JPEGImages/%s.jpg\n' % image_id)
            convert_annotation(image_id)
        list_file.close()