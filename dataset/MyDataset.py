# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 下午10:42
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
        self.gtbox_files = os.listdir(os.path.join(self.root, 'gtbox'))
        self.gtbox_files.sort()
        self.all_imgs = self.load_imgs()   #这里不能直接listdir，因为有的图片没有标注，应该根据标注数据来找出图片
        self.all_imgs.sort()
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_imgs(self):
        imgs = []
        for item in self.gtbox_files:
            imgs.append(item.split('.')[0] + '.jpg')
        return imgs

    def __getitem__(self, index):
        # 这里返回的是图片和标注文件的路径，因为图片大小和gtbox数量不一致，直接返回内容不大行
        gtbox = os.path.abspath(os.path.join(self.root, 'gtbox', self.gtbox_files[index]))
        img_path = os.path.join(self.root, 'imgs', self.all_imgs[index])
        return img_path, gtbox

    def __len__(self):
        return len(self.gtbox_files)

    def load_pic(self, img_path):
        img = Image.open(img_path)
        img = self.transforms(img)
        return img


class MyTestSet(data.Dataset):
    def __init__(self):
        self.root = './test'
        self.gtbox_files = os.listdir(os.path.join(self.root, 'gtbox'))
        self.gtbox_files.sort()
        self.all_imgs = os.listdir(os.path.join(self.root, 'imgs')) # 这里直接要全部的图片
        self.all_imgs.sort()
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'imgs', self.all_imgs[index])
        return img_path

    def __len__(self):
        return len(self.gtbox_files)

    def load_pic(self, img_path):
        img = Image.open(img_path)
        img = self.transforms(img)
        return img
