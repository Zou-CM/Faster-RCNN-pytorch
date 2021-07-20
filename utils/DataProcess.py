# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 下午10:44
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : DataProcess.py

import xml.dom.minidom
import os
from Config import Config
import sys
import math
import numpy as np
import cv2
import torch
from torchvision.ops import nms

cfg = Config()

def loadXml(path):
    '''
    读取原始的XMLlabel文件
    :return:
    '''
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement

    #获取图片上所有的目标
    objs = root.getElementsByTagName("object")

    #获取这张图片的长宽
    w, h = root.getElementsByTagName("width")[0].childNodes[0].data, root.getElementsByTagName("height")[0].childNodes[0].data

    #把所有目标的信息放到infos中返回
    infos = []

    for item in objs:
        c = item.getElementsByTagName("name")[0].childNodes[0].data
        xmin = item.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = item.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = item.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = item.getElementsByTagName("ymax")[0].childNodes[0].data
        # 数据集中有浮点数，先字符串转float再转int
        infos.append((cfg.cls_label.index(c), int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))))
    return np.array(infos)



def genDataset():
    '''
    把原始的数据集按照选取的模型的缩放比来调整，图片不用改，每张图片新建一个label文件和bbox文件，label文件对应的是anchor的label，
    :return:
    '''
    print("开始构建训练集的label和bbox文件")
    gtbox_path = '../train/gtbox'
    anno_path = '../train/Annotations'
    if not os.path.exists(gtbox_path):
        os.mkdir(gtbox_path)
    ann_list = os.listdir(anno_path)
    num = 1
    total = len(ann_list)
    for ann in ann_list:
        bar('正在构建：', num, total)
        num += 1
        name = ann.split('.')[0]
        gtbox_file = os.path.join(gtbox_path, name + '.npy')
        infos = loadXml(os.path.join(anno_path, ann))
        np.save(gtbox_file, infos)
    print('\n构建结束')
    print("开始构建测试集的label和bbox文件")
    gtbox_path = '../test/gtbox'
    anno_path = '../test/Annotations'
    if not os.path.exists(gtbox_path):
        os.mkdir(gtbox_path)
    ann_list = os.listdir(anno_path)
    num = 1
    total = len(ann_list)
    for ann in ann_list:
        bar('正在构建：', num, total)
        num += 1
        name = ann.split('.')[0]
        gtbox_file = os.path.join(gtbox_path, name + '.npy')
        infos = loadXml(os.path.join(anno_path, ann))
        np.save(gtbox_file, infos)
    print('\n构建结束')


def bar(msg, n, l):
    '''
    用来展示一下运行的进度的小功能
    :param msg:描述正在运行的进度
    :param n:运行的总数
    :param l:目前运行了多少
    :return:
    '''
    per = int(n*100/l)
    sys.stdout.write('\r' + msg + '[' + '*' * per + ' ' * (100 - per) + ']' + str(per) + '%')
    sys.stdout.flush()


if __name__ == '__main__':
    genDataset()

