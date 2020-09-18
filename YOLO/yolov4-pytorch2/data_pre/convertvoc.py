# -*- coding: utf-8 -*-

import shutil
import os
import json
import cv2

headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

# 上面的不用改


def writexml(idx, head, bbxes, tail):
    filename = ("../VOCdevkit_lb/VOC0807/Annotations/%06d.xml" % (idx))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % (bbx[0], bbx[1], bbx[2], bbx[3], bbx[4]))
        # 这里就是将文件中标签类别，左上角和右下角坐标存进去
    f.write(tail)
    f.close()

# 这个函数不用改
def clear_dir():
    if shutil.os.path.exists('../VOCdevkit_lb/VOC0807/Annotations'):
        shutil.rmtree('../VOCdevkit_lb/VOC0807/Annotations')
    if shutil.os.path.exists('../VOCdevkit_lb/VOC0807/ImageSets'):
        shutil.rmtree('../VOCdevkit_lb/VOC0807/ImageSets')
    if shutil.os.path.exists('../VOCdevkit_lb/VOC0807/JPEGImages'):
        shutil.rmtree('../VOCdevkit_lb/VOC0807/JPEGImages')

    shutil.os.mkdir('../VOCdevkit_lb/VOC0807/Annotations')
    shutil.os.makedirs('../VOCdevkit_lb/VOC0807/ImageSets/Main')
    shutil.os.mkdir('../VOCdevkit_lb/VOC0807/JPEGImages')


def excute_datasets(json_path, tr, idx):    # tr: train / val
    json_path = os.path.join(json_path, tr)
    json_file = os.listdir(json_path)   # 读取文件夹下的所有json文件
    savename = open(('../VOCdevkit_lb/VOC0807/ImageSets/Main/' + tr + '.txt'), 'a')         # 写入图片名字
    for file in json_file:
        file_path = os.path.join(json_path, file)   # 找到json文件路径
        with open(file_path, 'r', encoding='utf-8') as f:   # 开始读取json文件
            file_json = json.load(f)
            # imagename = file_json["imagePath"].split('\\')[-1]  # 找到当前json文件对应的图片名字
            imagename = file_json["imagePath"]  # 找到当前json文件对应的图片名字
            image_path = os.path.join('./image', imagename)    # 找到图片的路径，这里根据不一样的情况，也不一样，需要改
            image = cv2.imread(image_path)
            if image is None:   # 如果没有这种照片，跳过
                continue
            # label_shape_type = file_json['shapes'][0]['shape_type']
            # if label_shape_type != 'rectangle':     # 暂时不考虑其它形状的标签，在labelme这个软件打标签的时候有rectangle,还有多边形圆形这些，这些需要另外处理
            #     continue
            head = headstr % (idx, image.shape[1], image.shape[0], image.shape[2])
            # 这里是把对应图片路径，图片长，维度，宽存进去
            shapes = file_json['shapes']    # 这里是存储标签和坐标的位置，是一个列表，列表里面有很多字典，每个字典就代表一个标签
            boxes = []
            for i in range(len(shapes)):
                classname = file_json['shapes'][i]['label']     # 类别
                '''接下来转化类别为英文,因为labelme在标注的时候，为了标注人员的遍历，类别是中文，但是我们训练模型的时候必须是英文，
                这里就需要转化了，这里得xxxx不代表真的是xxxx，是你自己训练的类别，为了不让我老板看出我做的是他的项目，这里隐藏了'''
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'
                # if 'xxxxxxx' in classname:
                #     classname = 'xxxxxxx'

                box = [classname, file_json['shapes'][i]['points'][0][0], file_json['shapes'][i]['points'][0][1],
                       file_json['shapes'][i]['points'][2][0], file_json['shapes'][i]['points'][2][1]]  # 这里存储的意义是[标签类别， x1,y1, x2, y2]

                boxes.append(box)

            writexml(idx, head, boxes, tailstr)
            cv2.imwrite('../VOCdevkit_lb/VOC0807/JPEGImages/%06d.jpg' % (idx), image)
            savename.write('%06d\n' % idx)  # 写入图片编号
            idx += 1
    savename.close()
    return idx


if __name__ == '__main__':
    clear_dir()
    idx = 1
    idx = excute_datasets('./json', 'train', idx)
    idx = excute_datasets('./json', 'val', idx)
    print('Complete...')
