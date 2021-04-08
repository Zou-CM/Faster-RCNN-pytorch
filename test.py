# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 下午1:05
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : test.py


from dataset.MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import numpy as np

testSet = MyDataset('test')
testLoader = DataLoader(testSet, batch_size=1, shuffle=True, drop_last=True)
for n, data in enumerate(testLoader):
    _, boxes_root, _ = data
    boxes = []
    for root in boxes_root:
        boxes.extend(testSet.load_box_or_label(root))
    boxes = np.array(boxes)
    np.reshape(boxes, (-1, 11))
    # print(boxes)
    boxes = torch.from_numpy(boxes)
    boxes = Variable(boxes)
    # boxes = boxes.cuda()
    labels = boxes[:, -1]
    labels = labels.cuda()
    index = [index for index in range(len(labels)) if labels[index] != -1]
    print()
