# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : DataProcess.py

import xml.dom.minidom
import os
from Config import Config
import sys
import math
import numpy as np

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
        infos.append((c, int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))))
    return int(w), int(h), infos

def genBbox(path, w, h, infos):
    '''
    保存修改后的bbox坐标
    :param path: 保存的路径
    :param w: 原图宽
    :param h: 原图高
    :param infos: 图中所有目标的信息
    :return: new_infos缩放后的bbox信息
    '''
    #获取缩放因子
    factor = cfg.min_size * 1.0 / min(w, h)
    new_infos = []
    for item in infos:
        cls = item[0]
        xmin = int(item[1] * 1.0 * factor)
        ymin = int(item[2] * 1.0 * factor)
        xmax = int(item[3] * 1.0 * factor)
        ymax = int(item[4] * 1.0 * factor)
        x = math.floor((xmax + xmin) / 2)
        y = math.floor((ymax + ymin) / 2)
        gw = xmax - xmin + 1
        gh = ymax - ymin + 1
        cls = cfg.cls_label.index(cls)
        info = [cls, x, y, gw, gh]
        new_infos.append(info)
    np.save(path, np.array(new_infos))
    return new_infos

def calcIOU(box1, box2):
    '''
    计算两个框的iou
    :param box1:框1
    :param box2: 框2
    :return:
    '''
    xa, ya, wa, ha = box1
    xb, yb, wb, hb = box2
    x1 = xa - wa // 2
    y1 = ya - ha // 2
    x2 = x1 + wa
    y2 = y1 + ha
    x3 = xb - wb // 2
    y3 = yb - hb // 2
    x4 = x3 + wb
    y4 = y3 + hb
    xmin = max(x1, x3)
    ymin = max(y1, y3)
    xmax = min(x2, x4)
    ymax = min(y2, y4)
    if xmin >= xmax or ymin >= ymax:
        return 0.0
    if (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-(xmax-xmin)*(ymax-ymin) == 0:
        return 1.0
    return (xmax-xmin)*(ymax-ymin)*1.0/((x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-(xmax-xmin)*(ymax-ymin))



def checkLabel(anchors, w, h, infos):
    '''
    给anchor确定label
    :param anchors: 图片上的所有anchor信息
    :param w: 图片宽（已经缩放过）
    :param h: 图片高（已经缩放过）
    :param infos: 目标框信息
    :return: 直接在anchors上添加label
    '''
    # 记录一下还没有标记label的anchor的下标
    Set = set(range(len(anchors)))

    # 记录正样本anchor的下标
    Set_pos = set()

    # 记录负样本anchor的下标
    Set_neg = set()

    # 标记超出图片范围的anchor
    for i in range(len(anchors)):
        ax, ay, aw, ah = anchors[i]
        if ax - aw // 2 < 0 or ay - ah // 2 < 0 or ax + aw // 2 - 1 > w or ay + ah // 2 - 1 > h:
            anchors[i].extend([-1, -1, -1, -1])
            anchors[i].append(-1)
            Set.remove(i)
    # 标记每个目标框对应的最大iou的anchor
    for info in infos:
        _, gx, gy, gw, gh = info
        tmp_iou = -1
        index = -1
        for i in Set:
            ax, ay, aw, ah = anchors[i]
            iou = calcIOU((gx, gy, gw, gh), (ax, ay, aw, ah))
            if iou != 0 and iou > tmp_iou:
                index = i
        if index != -1:
            anchors[index].extend([gx, gy, gw, gh])
            anchors[index].append(1)
            Set.remove(index)
            Set_pos.add(index)
    # 根据iou标记剩余anchor
    for i in Set:
        ax, ay, aw, ah = anchors[i]
        flag = -1
        max_iou = -1
        bbox = None
        for info in infos:
            _, gx, gy, gw, gh = info
            iou = calcIOU((gx, gy, gw, gh), (ax, ay, aw, ah))
            # print(xmin, ymin, xmax, ymax, x1, y1, x2, y2, iou)
            if iou >=0.7 and iou > max_iou:
                flag = 1
                bbox = [gx, gy, gw, gh]
            elif iou >= 0.3:
                flag = 0
        # 大于0.7正样本，小于0.3负样本，其余忽略
        if flag == 1:
            Set_pos.add(i)
            anchors[i].extend(bbox)
            anchors[i].append(1)
        elif flag == 0:
            anchors[i].extend([-1, -1, -1, -1])
            anchors[i].append(-1)
        else:
            Set_neg.add(i)
            anchors[i].extend([-1, -1, -1, -1])
            anchors[i].append(0)

    ####这是原文的做法，但我发现正样本也太少了，就改了改
    # # 控制正负样本的量和比例
    # delta = len(Set_pos) - 128
    # # if delta > 0:
    # #     print(len(Set_pos))
    # if delta > 0:
    #     list_pos = list(Set_pos)
    #     for i in range(delta):
    #         anchors[list_pos[i]][-1] = -1
    # # 保证正样本不足128时，用负样本补全256个
    # # print(len(Set_neg))
    # delta = len(Set_neg) - 128 - delta
    # if delta > 0:
    #     list_neg = list(Set_neg)
    #     for i in range(delta):
    #         anchors[list_neg[i]][-1] = -1

    # 我控制正样本在128个以内，但不足不用负样本补全，负样本个数和正样本保持一致即可，不然正负比也太失衡了
    # 控制正负样本的量和比例
    delta = len(Set_pos) - 128
    if delta > 0:
        print(len(Set_pos))
        list_pos = list(Set_pos)
        for i in range(delta):
            anchors[list_pos[i]][-1] = -1
            Set_pos.remove(list_pos[i])
    # 保证正样本不足128时，用负样本补全256个
    # print(len(Set_neg))
    delta = len(Set_neg) - len(Set_pos)
    if delta > 0:
        list_neg = list(Set_neg)
        for i in range(delta):
            anchors[list_neg[i]][-1] = -1


def genRC(width, heigth):
    '''
    用于求不用pool的模型
    :param width:
    :param heigth:
    :return:
    '''
    for i in range(4):
        width = (width - 1) // 2 + 1
        heigth = (heigth - 1) // 2 + 1
    return width, heigth



def genLabel(path, w, h, infos):
    '''
    获取每张图片anchor的labels
    :param path: 保存的路径
    :param w: 原图宽
    :param h: 原图高
    :param infos: 图中所有目标的信息
    :return:
    '''
    # 获取缩放因子
    factor = cfg.min_size * 1.0 / min(w, h)

    # 缩放后的图片尺寸
    width, height = int(w * factor), int(h * factor)

    # 获取特征图层的尺寸
    r, c = genRC(width, height)
    anchors = []
    for i in range(r):
        for j in range(c):
            # 获取每个anchor的中心点坐标
            y = 8 * (2 * i + 1)
            x = 8 * (2 * j + 1)
            for (aw, ah) in cfg.anchor_size:
                # 获取anchor的两点坐标
                anchors.append([x, y, aw, ah])
    checkLabel(anchors, width, height, infos)
    anchors = np.array(anchors)
    np.save(path, anchors)
    # with open(path, 'w') as fw:
    #     for anchor in anchors:
    #         line = ','.join(map(str, anchor)) + '\n'
    #         fw.write(line)



def genDataset():
    '''
    把原始的数据集按照选取的模型的缩放比来调整，图片不用改，每张图片新建一个label文件和bbox文件，label文件对应的是anchor的label，
    :return:
    '''
    print("开始构建训练集的label和bbox文件")
    label_path = '../train/labels'
    bbox_path = '../train/bboxs'
    anno_path = '../train/Annotations'
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    if not os.path.exists(bbox_path):
        os.mkdir(bbox_path)
    ann_list = os.listdir(anno_path)
    num = 1
    total = len(ann_list)
    for ann in ann_list:
        bar('正在构建：', num, total)
        num += 1
        name = ann.split('.')[0]
        label_file = os.path.join(label_path, name + '.npy')
        bbox_file = os.path.join(bbox_path, name + '.npy')
        w, h, infos = loadXml(os.path.join(anno_path, ann))
        new_infos = genBbox(bbox_file, w, h, infos)
        genLabel(label_file, w, h, new_infos)
    print('\n构建结束')
    print("开始构建测试集的label和bbox文件")
    label_path = '../test/labels'
    bbox_path = '../test/bboxs'
    anno_path = '../test/Annotations'
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    if not os.path.exists(bbox_path):
        os.mkdir(bbox_path)
    ann_list = os.listdir(anno_path)
    num = 1
    total = len(ann_list)
    for ann in ann_list:
        bar('正在构建：', num, total)
        num += 1
        name = ann.split('.')[0]
        label_file = os.path.join(label_path, name + '.npy')
        bbox_file = os.path.join(bbox_path, name + '.npy')
        w, h, infos = loadXml(os.path.join(anno_path, ann))
        new_infos = genBbox(bbox_file, w, h, infos)
        genLabel(label_file, w, h, new_infos)
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

