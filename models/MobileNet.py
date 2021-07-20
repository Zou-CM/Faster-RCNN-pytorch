# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : MobileNet.py

from torchvision.models import MobileNetV2
import torch.nn as nn
import torch.functional as F



class mobilenet_v2_x16(nn.Module):
    def __init__(self):
        super(mobilenet_v2_x16, self).__init__()
        self.inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 1], #原本这层stride=2
                [6, 320, 1, 1],
            ]
        self.feature_layer = MobileNetV2(inverted_residual_setting=self.inverted_residual_setting).features

    def forward(self, x):
        return self.feature_layer(x)
