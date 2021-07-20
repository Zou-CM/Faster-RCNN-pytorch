# -*- coding: utf-8 -*-
# @Time    : 2021/5/9 下午1:10
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : eval.py

import torch
from Config import Config
from torch.utils.data import DataLoader
from dataset.MyDataset import MyTestSet
from torch.autograd import Variable
from torchvision.ops import nms
import cv2
import os
from utils.DataProcess import bar

def eval():
    cfg = Config()

    net = torch.load(cfg.FasterRCNN_checkpoints)

    print(net)

    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    testset = MyTestSet()

    testLoader = DataLoader(testset, batch_size=1, shuffle=True, drop_last=True)

    total_test = len(testLoader)

    for n, data in enumerate(testLoader):
        img_path = data[0]
        img_name = img_path.split('/')[-1].split('.')[0]
        img = testset.load_pic(img_path)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        res = net([img])
        res = res[0]
        boxes = res["boxes"]
        labels = res["labels"]
        scores = res["scores"]
        index = torch.where(scores >= 0.5)
        boxes = boxes[index]
        labels = labels[index]
        scores = scores[index]
        index = nms(boxes, scores, 0.7)
        boxes = boxes[index]
        labels = labels[index]
        boxes = boxes.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        img = cv2.imread(img_path)
        ori_path = os.path.join(cfg.dev_path, img_name + '.jpg')
        cv2.imwrite(ori_path, img)
        for i in range(len(boxes)):
            if labels[i] == 0:
                continue
            cv2.rectangle(img, (int(boxes[i, 0]), int(boxes[i, 1])), (int(boxes[i, 2]), int(boxes[i, 3])),
                          color=(0, 0, 255), thickness=1)
            cv2.putText(img, cfg.cls_label[labels[i]], (int(boxes[i, 0]), int(boxes[i, 1])+15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
        aft_path = os.path.join(cfg.dev_path, img_name + '_detect.jpg')
        cv2.imwrite(aft_path, img)
        bar("正在测试：", n+1, total_test)


if __name__ == '__main__':
    eval()