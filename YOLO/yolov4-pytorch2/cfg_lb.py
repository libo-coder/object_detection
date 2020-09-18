# -*- coding: utf-8 -*-
import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.use_darknet_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

Cfg.batch = 4      # 每次迭代要进行训练的图片数量
Cfg.subdivisions = 1       # 源码中的图片数量int imgs = net.batch * net.subdivisions * ngpus，按subdivisions大小分批进行训练
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.949        # 冲量
Cfg.decay = 0.0005          # 权值衰减
Cfg.angle = 0               # 图片角度变化，单位为度,假如angle=5，就是生成新图片的时候随机旋转-5~5度
Cfg.saturation = 1.5        # 饱和度变化大小
Cfg.exposure = 1.5          # 曝光变化大小
Cfg.hue = .1                # 色调变化范围，tiny-yolo-voc.cfg中-0.1~0.1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500        # 训练次数，建议设置为classes*2000，但是不要低于4000
Cfg.steps = [400000, 450000]    # 一般设置为max_batch的80%与90%
Cfg.policy = Cfg.steps          # 调整学习率的策略
Cfg.scales = .1, .1             # 相对于当前学习率的变化比率，累计相乘，与steps中的参数个数保持一致；

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 1
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 300
Cfg.train_label = os.path.join(_BASE_DIR, 'VOCdevkit_lb/VOC0807/ImageSets/Main', 'train.txt')
Cfg.val_label = os.path.join(_BASE_DIR, 'VOCdevkit_lb/VOC0807/ImageSets/Main','val.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10
