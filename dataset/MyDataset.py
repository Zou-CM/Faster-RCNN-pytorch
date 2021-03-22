# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 下午10:47
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : MyDataset.py


from torch.utils import data
import os

class MyDataset(data.Dataset):
    def __init__(self, root):
        self.root = '../' + root   #root为'train'或者'test'
        self.label_files = os.listdir(os.path.join(self.root, 'labels'))
        self.bbox_files = os.listdir(os.path.join(self.root, 'bboxs'))
        self.all_imgs = self.load_imgs()   #这里不能直接listdir，因为有的图片没有标注，应该根据标注数据来找出图片

    def load_imgs(self):
        imgs = []
        for item in self.label_files:
            imgs.append(item.split('.')[0] + '.jpg')
        return imgs

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.label_files)