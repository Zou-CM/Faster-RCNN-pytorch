# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : RPN.py

from torchvision.models import mobilenet_v2
import torchvision.models as models
from models.resnet34 import resnet34
import torch.nn as nn
import torch
import numpy as np
import torch.functional as F
from Config import Config
import time

cfg = Config()
from utils.LossFunction import *


class RPN(nn.Module):
    def __init__(self, pretrained):
        super(RPN, self).__init__()
        # self.base_net = mobilenet_v2(pretrained).features
        # self.feature_layer = nn.Sequential(
        #     self.base_net[0],
        #     self.base_net[1],
        #     self.base_net[2],
        #     self.base_net[3],
        #     self.base_net[4],
        #     self.base_net[5],
        #     self.base_net[6],
        #     self.base_net[7],
        #     self.base_net[8],
        #     self.base_net[9],
        #     self.base_net[10],
        #     self.base_net[11],
        #     self.base_net[12],
        #     self.base_net[13]
        # )
        self.feature_layer = resnet34(pretrained)
        self.conv_1 = nn.Conv2d(cfg.feature_map_channels, 256, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(256, 9 * 2, kernel_size=1)
        self.conv_box = nn.Conv2d(256, 9 * 4, kernel_size=1)

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.conv_1(x)
        x = nn.ReLU()(x)
        # reshape是为了更方便计算损失
        cls = self.conv_cls(x)
        cls = cls.permute(0, 2, 3, 1)
        cls = torch.reshape(cls, (-1, 2))
        box = nn.ReLU()(self.conv_box(x))
        box = box.permute(0, 2, 3, 1)
        box = torch.reshape(box, (-1, 4))
        return cls, box

    def loss_cls(self, labels, cls):
        '''
        计算前景分类损失
        :param labels: anchor的label
        :param cls: RPN出来的label
        :return:
        '''
        # 排除忽略的anchor
        index = torch.where(labels > -1)
        lab = labels[index]
        cl = cls[index]
        return nn.CrossEntropyLoss()(cl, lab)

    def loss_box(self, labels, anchors, gt_box, pre_box):
        '''
        计算坐标回归的损失
        :param labels: anchor的label
        :param anchors:原始的anchor
        :param gt_box: 图中的bbox坐标
        :param pre_box: 回归的结果坐标
        :return:
        '''
        # 只对正样本进行回归计算
        index = torch.where(labels == 1)
        an = anchors[index]
        gt = gt_box[index]
        pr = pre_box[index]
        # 预测框的中心点坐标和长宽，这里直接预测中心点和w，h，如果还是拟合对角两点的话，有可能w或h小于零
        w = pr[:, 0].float()
        h = pr[:, 1].float()
        x = pr[:, 2].float()
        y = pr[:, 3].float()
        # anchor的中心点坐标和长宽
        wa = an[:, 2].float()
        ha = an[:, 3].float()
        xa = an[:, 0].float()
        ya = an[:, 1].float()
        # bbox的中心点坐标和长宽
        wb = gt[:, 2].float()
        hb = gt[:, 3].float()
        xb = gt[:, 0].float()
        yb = gt[:, 1].float()
        # 预测框和anchor的偏差
        tx = (x - xa) * 1.0 / wa
        ty = (y - ya) * 1.0 / ha
        tw = torch.log(w * 1.0 / wa + 1)  # 避免w为0时tw为-inf
        th = torch.log(h * 1.0 / ha + 1)
        # bbox和anchor的偏差
        t_x = (xb - xa) * 1.0 / wa
        t_y = (yb - ya) * 1.0 / ha
        t_w = torch.log(wb * 1.0 / wa + 1)
        t_h = torch.log(hb * 1.0 / ha + 1)
        # print(t_x,tx,t_y,ty,t_w,tw,t_h,th)
        # print(x,y,h,w,wa,ha)
        return smoothL1(tx, t_x) + smoothL1(ty, t_y) + smoothL1(tw, t_w) + smoothL1(th, t_h)
