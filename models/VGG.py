# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 下午10:22
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : VGG.py


import torchvision.models as models
import torch.nn as nn
import torch.functional as F


net = models.vgg19(False)
net.features[-1] = nn.Sequential()
print(net)
