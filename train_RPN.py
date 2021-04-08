# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午8:37
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : train_RPN.py


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from dataset.MyDataset import MyDataset
from torchvision import models
import os
from Config import Config
from models.RPN import RPN
from utils.DataProcess import bar
import numpy as np
import time


def train(pretrained):
    '''
    训练RPN网络
    :param pretrained:第一次训练应该load主干网络的预训练模型
    :return:
    '''
    cfg = Config()

    net = RPN(pretrained=pretrained)
    print(net)
    net.cuda()

    if pretrained == False:
        net.load_state_dict(torch.load(cfg.RPN_checkpoints))

    trainSet = MyDataset('train')
    testSet = MyDataset('test')

    trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, drop_last=True)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=True, drop_last=True)

    # for name, value in net.named_parameters():
    #     if 'feature_layer' in name:
    #         value.requires_grad = False
    #
    # params = filter(lambda p: p.requires_grad, net.parameters())

    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    total_train = len(trainLoader) - 1

    for epoch in range(1, cfg.epoch_RPN+1):
        net.train()
        if epoch % 6 == 0:
            for p in opt.param_groups:
                p['lr'] *= 0.1
        for n, data in enumerate(trainLoader):
            img, boxes_root, _ = data
            boxes = np.load(boxes_root[0])
            np.reshape(boxes, (-1, 9))
            # print(boxes)
            boxes = torch.from_numpy(boxes)
            boxes = Variable(boxes)
            boxes = boxes.cuda()
            anchors = boxes[:, 0:4]
            gt_boxes = boxes[:, 4:8]
            labels = boxes[:, -1]
            # anchors = anchors.cuda()
            # gt_boxes = gt_boxes.cuda()
            # labels = labels.cuda()
            img = Variable(img)
            img = img.cuda()
            # anchors = Variable(anchors)
            # gt_boxes = Variable(gt_boxes)
            # labels = Variable(labels)
            cls, box = net(img)
            opt.zero_grad()
            cls_loss = net.loss_cls(labels, cls)
            box_loss = net.loss_box(labels, anchors, gt_boxes, box)
            # print(cls_loss.data, box_loss.data)
            loss = cls_loss + 10*box_loss
            # loss = cls_loss
            # print(loss)
            loss.backward()
            opt.step()
            bar('正在第%d轮训练,loss_label=%.5f,loss_box=%.5f,total_loss=%.5f'%(epoch, cls_loss.data,box_loss.data,loss.data), n, total_train)
        net.eval()
        acc = 0.0
        test_num = 0.0
        # 这里只是校验一下前景识别的准确率，没有计算bbox的损失
        for n, data in enumerate(testLoader):
            if n == 1000:
                break
            img, boxes_root, _ = data
            boxes = np.load(boxes_root[0])
            np.reshape(boxes, (-1, 9))
            # print(boxes)
            boxes = torch.from_numpy(boxes)
            boxes = Variable(boxes)
            boxes = boxes.cuda()
            labels = boxes[:, -1]
            # labels = labels.cuda()
            img = Variable(img)
            img = img.cuda()
            cls, _ = net(img)
            logits = torch.argmax(cls, dim=1)
            # print(logits)
            index = [index for index in range(len(labels)) if labels[index] != -1]
            logits = logits[index]
            labels = labels[index]
            test_num += len(labels)
            for i in range(len(labels)):
                if logits[i] == labels[i]:
                    acc += 1
        print('\n第%d轮训练测试准确率为：%.3f'%(epoch, acc / test_num))
        torch.save(net.state_dict(), cfg.RPN_checkpoints)






if __name__ == '__main__':
    train(True)