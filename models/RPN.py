# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : RPN.py

from MobileNet import MobileNet
import torch.nn as nn
import torch
import numpy as np
import torch.functional as F
from Config import Config
cfg = Config()
from utils.LossFunction import smoothL1

class RPN(nn.Module):
    def __init__(self, pretrained):
        super(RPN, self).__init__()
        self.feature_layer = MobileNet(pretrained)
        self.conv_1 = nn.Conv2d(cfg.feature_map_channels, 512, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(512, 2, kernel_size=1)
        self.conv_box = nn.Conv2d(512, 9*4, kernel_size=1)

    def forward(self, x):
        x = self.feature_layer(x)
        x = nn.BatchNorm2d(512)(self.conv_1(x))
        x = nn.ReLU()(x)
        # reshape是为了更方便计算损失
        cls = nn.ReLU()(self.conv_cls(x))
        cls = torch.reshape(cls, (-1, 2))
        box = nn.ReLU()(self.conv_box(x))
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
        index = [i for i in range(len(labels)) if labels[i] != -1]
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
        index = [i for i in range(len(labels)) if labels[i] == 1]
        an = anchors[index]
        gt = gt_box[index]
        pr = pre_box[index]
        # 预测框的中心点坐标和长宽
        w = (pr[:, 2] - pr[:, 0]).astype(int)
        h = (pr[:, 3] - pr[:, 1]).astype(int)
        x = ((pr[:, 2] + pr[:, 0]) / 2).astype(int)
        y = ((pr[:, 3] + pr[:, 1]) / 2).astype(int)
        # anchor的中心点坐标和长宽
        wa = (an[:, 2] - an[:, 0]).astype(int)
        ha = (an[:, 3] - an[:, 1]).astype(int)
        xa = ((an[:, 2] + an[:, 0]) / 2).astype(int)
        ya = ((an[:, 3] + an[:, 1]) / 2).astype(int)
        # bbox的中心点坐标和长宽
        wb = (gt[:, 2] - gt[:, 0]).astype(int)
        hb = (gt[:, 3] - gt[:, 1]).astype(int)
        xb = ((gt[:, 2] + gt[:, 0]) / 2).astype(int)
        yb = ((gt[:, 3] + gt[:, 1]) / 2).astype(int)
        # 预测框和anchor的偏差
        tx = (x - xa) * 1.0 / wa
        ty = (y - ya) * 1.0 / ha
        tw = torch.log(w * 1.0 / wa)
        th = torch.log(h * 1.0 / ha)
        # bbox和anchor的偏差
        t_x = (xb - xa) * 1.0 / wa
        t_y = (yb - ya) * 1.0 / ha
        t_w = torch.log(wb * 1.0 / wa)
        t_h = torch.log(hb * 1.0 / ha)
        return smoothL1(tx, t_x) + smoothL1(ty, t_y) + smoothL1(tw, t_w) + smoothL1(th, t_h)




