# -*- coding: utf-8 -*-
# @Time    : 2021/4/1 下午9:58
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : resnet34.py

import torchvision.models as models
import torch.nn as nn

class resnet34(nn.Module):
    def __init__(self, pretrained):
        super(resnet34, self).__init__()
        self.base_net = models.resnet34(pretrained)
        self.feature_layer = nn.Sequential(
            self.base_net.conv1,
            self.base_net.bn1,
            self.base_net.relu,
            self.base_net.maxpool,
            self.base_net.layer1,
            self.base_net.layer2,
            self.base_net.layer3
        )

    def forward(self, x):
        x = self.feature_layer(x)
        return x
