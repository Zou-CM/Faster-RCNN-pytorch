# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 下午10:47
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : MyDataset.py


from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from Config import Config

cfg = Config()

class MyDataset(data.Dataset):
    def __init__(self, root):
        self.root = './' + root   #root为'train'或者'test'
        self.label_files = os.listdir(os.path.join(self.root, 'labels'))
        self.label_files.sort()
        self.bbox_files = os.listdir(os.path.join(self.root, 'bboxs'))
        self.bbox_files.sort()
        self.all_imgs = self.load_imgs()   #这里不能直接listdir，因为有的图片没有标注，应该根据标注数据来找出图片
        self.all_imgs.sort()
        self.transforms = transforms.Compose([
            transforms.Resize(cfg.min_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 简单归一化一下到[-1, 1]，不是很专业
        ])

    def load_imgs(self):
        imgs = []
        for item in self.label_files:
            imgs.append(item.split('.')[0] + '.jpg')
        return imgs

    def __getitem__(self, index):
        labels = os.path.abspath(os.path.join(self.root, 'labels', self.label_files[index]))
        boxes = os.path.abspath(os.path.join(self.root, 'bboxs', self.bbox_files[index]))
        img = Image.open(os.path.join(self.root, 'imgs', self.all_imgs[index]))
        img = self.transforms(img)
        return img, labels, boxes

    def __len__(self):
        return len(self.label_files)

    def load_box_or_label(self, path):
        ans = []
        with open(path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                ans.append(list(map(int, line.strip().split(','))))
        return ans
