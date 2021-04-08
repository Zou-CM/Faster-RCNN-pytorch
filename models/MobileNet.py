# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : MobileNet.py

from torchvision.models.mobilenet import MobileNetV2
import torch.nn as nn
import torch.functional as F



class MobileNet(MobileNetV2):
    def __init__(self, pretrained):
        super(MobileNet, self).__init__()

    def _forward_impl(self, x):
        return self.feature(x)


net = MobileNet(False)
print(net)
