# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 下午9:49
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : resnet101.py

import torchvision.models as models
import torch.nn as nn

class resnet101_x16(nn.Module):
    def __init__(self, pretrained):
        super(resnet101_x16, self).__init__()
        self.base_net = models.resnet101(pretrained)
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


if __name__ == '__main__':
    net = resnet101_x16(False)
    print(net)