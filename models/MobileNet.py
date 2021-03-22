# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : MobileNet.py

from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.functional as F

class MobileNet(nn.Module):
    def __init__(self, pretrained):
        super(MobileNet, self).__init__()
        self.feature = mobilenet_v2(pretrained=pretrained)
        # 把全链接分类那一块置空
        self.feature.classifier = nn.Sequential()

    def forward(self, x):
        return self.feature(x)


net = MobileNet(False)
print(net)