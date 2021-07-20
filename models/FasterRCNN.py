# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 下午9:37
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : FasterRCNN.py

from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from Config import Config
from models.VGG import VGG16
from models.resnet34 import resnet34
from torchvision.models import mobilenet_v2
from models.resnet101 import resnet101_x16
from models.MobileNet import mobilenet_v2_x16


cfg = Config()


class FasterRCNN:
    def __init__(self):
        self.net = self.initNet()

    def initNet(self):
        # 确定backbone
        backbone = None
        if cfg.backbone == "VGG":
            backbone = VGG16(True)
            backbone.out_channels = 512
        elif cfg.backbone == "resnet34":
            backbone = resnet34(True)
            backbone.out_channels = 256
        elif cfg.backbone == "resnet101_x16":
            backbone = resnet101_x16(True)
            backbone.out_channels = 1024
        elif cfg.backbone == "mobilenet_V2":
            backbone = mobilenet_v2(True).features
            backbone.out_channels = 1280
        elif cfg.backbone == "mobilenet_V2_x16":
            backbone = mobilenet_v2_x16()
            backbone.out_channels = 1280

        # 生成anchor
        anchor_generator = AnchorGenerator(sizes=cfg.anchor_size, aspect_ratios=cfg.aspect_ratios)

        # 设置roipooling，采用ROIAlign
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # 建立FasterRCNN模型
        net = faster_rcnn.FasterRCNN(backbone,
                                     num_classes=cfg.class_num+1,
                                     rpn_anchor_generator=anchor_generator,
                                     box_roi_pool=roi_pooler)
        return net


