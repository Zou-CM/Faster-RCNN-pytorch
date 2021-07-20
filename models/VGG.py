# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 下午10:22
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : VGG.py


import torchvision.models as models
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, pretrained):
        super(VGG16, self).__init__()
        self.base_net = models.vgg16(pretrained).features
        self.base_net[30] = nn.Sequential()

    def forward(self, x):
        return self.base_net(x)

if __name__ == '__main__':
    net = VGG16(False)
    print(net)

