#encoding=utf-8

import xml.dom.minidom
import os
from Config import Config
import sys

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
    infos = []
    #把所有目标的信息放到infos中返回
    for item in objs:
        c = item.getElementsByTagName("name")[0].childNodes[0].data
        xmin = item.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = item.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = item.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = item.getElementsByTagName("ymax")[0].childNodes[0].data
        infos.append((c, int(xmin), int(ymin), int(xmax), int(ymax)))
    return int(w), int(h), infos

def genBbox(path, w, h, infos):
    '''
    保存修改后的bbox坐标
    :param path: 保存的路径
    :param w: 原图宽
    :param h: 原图长
    :param infos: 图中所有目标的信息
    :return:
    '''
    #获取缩放因子
    factor = cfg.min_size * 1.0 / min(w, h)
    with open(path, 'w') as fw:
        for item in infos:
            cls = item[0]
            xmin = int(item[1] * 1.0 * factor)
            ymin = int(item[2] * 1.0 * factor)
            xmax = int(item[3] * 1.0 * factor)
            ymax = int(item[4] * 1.0 * factor)
            info = [cls, xmin, ymin, xmax, ymax]
            line = ','.join(map(str, info)) + '\n'
            fw.write(line)



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
        bar('', num, total)
        num += 1
        name = ann.split('.')[0]
        label_file = os.path.join(label_path, name + '.txt')
        bbox_file = os.path.join(bbox_path, name + '.txt')
        w, h, infos = loadXml(os.path.join(anno_path, ann))
        genBbox(bbox_file, w, h, infos)


def bar(msg, n, l):
    '''
    用来展示一下运行的进度的小功能
    :param msg:描述正在运行的进度
    :param n:运行的总数
    :param l:目前运行了多少
    :return:
    '''
    per = int(n*100/l)
    sys.stdout.write('\r' + msg + '[' + '*' * per + ' ' * (100 - per) + ']')
    sys.stdout.flush()

if __name__ == '__main__':
    genDataset()

