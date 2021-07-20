# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : Config.py

class Config():
    # anchor的size和长宽比
    anchor_size = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1, 2),)
    # 特征网络
    backbone = "mobilenet_V2"
    # 目标类型，0是背景
    cls_label = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                 'tvmonitor', 'cat'] #VOC所有标签
    class_num = 20
    lr = 0.0001
    epoch = 6
    batch_size = 2
    # 模型保存位置
    FasterRCNN_checkpoints = './checkpoints/FasterRCNN/model.pth'
    # 校验图像保存位置
    dev_path = '/home/zcm/deeplearning/Faster-RCNN/dev_img'