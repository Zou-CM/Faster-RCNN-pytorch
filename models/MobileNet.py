#encoding=utf-8

from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.functional as F

class MobileNet(nn.Module):
    def __init__(self, pretrained):
        super(MobileNet, self).__init__()
        self.feature = mobilenet_v2(pretrained=pretrained)
        self.feature.classifier = nn.Sequential()

    def forward(self, x):
        return self.feature(x)


net = MobileNet(True)
print(net)