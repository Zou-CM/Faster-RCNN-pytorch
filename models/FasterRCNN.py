# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 下午9:56
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : FasterRCNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RPN import RPN
from Config import Config
from torchvision.ops import roi_pool
from torchvision.ops import nms, box_iou

cfg = Config()

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.rpn = RPN(False)
        self.feature_layer = self.rpn.feature_layer
        self.conv_1 = self.rpn.conv_1
        self.conv_box = self.rpn.conv_box
        self.conv_cls = self.rpn.conv_cls
        self.rpn.load_state_dict(torch.load(cfg.RPN_checkpoints))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_cls = nn.Linear(4096, cfg.class_num+1)
        self.fc_box = nn.Linear(4096, 4)

    def forward(self, x, w, h, gt_boxes):
        feature = self.feature_layer(x)
        x = self.conv_1(feature)
        x = F.relu(x)
        pre_labels = self.conv_cls(x)
        pre_boxes = self.conv_box(x)
        pre_labels = pre_labels.permute(0, 2, 3, 1)
        pre_labels = torch.reshape(pre_labels, (-1, 2))
        pre_boxes = pre_boxes.permute(0, 2, 3, 1)
        pre_boxes = torch.reshape(pre_boxes, (-1, 4))
        logits = torch.nn.functional.softmax(pre_labels) #softmax之后把前景维度得分定为预测框的得分
        logits = logits[:, 1]
        index = torch.where(logits>0.5) #只拿label是1的预测框进入下一步
        logits = logits[index]
        pre_boxes = pre_boxes[index]
        # 把宽高中心点形式转成两点式,超出范围的截取部分
        for i in range(len(pre_boxes)):
            x1 = max(pre_boxes[i, 2] - pre_boxes[i, 0] / 2, 0)
            y1 = max(pre_boxes[i, 3] - pre_boxes[i, 1] / 2, 0)
            x2 = min(pre_boxes[i, 2] + pre_boxes[i, 0], w)
            y2 = min(pre_boxes[i, 3] + pre_boxes[i, 1], h)
            pre_boxes[i, 0] = x1
            pre_boxes[i, 1] = y1
            pre_boxes[i, 2] = x2
            pre_boxes[i, 3] = y2
        # nms后得到保留的下标
        pre_boxes = pre_boxes[nms(pre_boxes, logits, 0.7)]
        pre_boxes = pre_boxes.unsqueeze(dim=0) # 增加一个batch维度，方便调用roipooling
        x = roi_pool(feature, pre_boxes, (7, 7), cfg.scale)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        cls = self.fc_cls(x)
        box = self.fc_box(x)
        labels = gt_boxes[:, 0] # 获取每个gt_box的label
        bbox_iou = box_iou(box, gt_boxes[:, 1:]) # 获取每个预测框和gt_box的iou



