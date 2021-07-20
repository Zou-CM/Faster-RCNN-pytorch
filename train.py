# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 下午10:26
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : train.py

from dataset.MyDataset import MyDataset
from models.FasterRCNN import FasterRCNN
from torch.utils.data import DataLoader
from Config import Config
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch
from utils.DataProcess import bar
from torch.utils.tensorboard import SummaryWriter


def train():
    # # 调用tensorboard监视训练过程
    # writer = SummaryWriter("runs/summary")
    cfg = Config()

    net = FasterRCNN().net

    net.cuda()

    trainset = MyDataset('train')

    trainLoader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    params = [p for p in net.parameters() if p.requires_grad]

    opt = optim.Adam(params, lr=cfg.lr)

    num_train = len(trainLoader)

    for epoch in range(1, cfg.epoch + 1):
        net.train()
        # 每2轮学习率下降一次
        if epoch % 2 == 0:
            for p in opt.param_groups:
                p['lr'] *= 0.1
        for n, data in enumerate(trainLoader):
            imgs = []
            gtbox = []
            for i in range(cfg.batch_size):
                img_path = data[0][i]
                gtbox_path = data[1][i]
                img = trainset.load_pic(img_path)
                img = Variable(img)
                gtbbox = np.load(gtbox_path)
                gtbbox = torch.from_numpy(gtbbox)
                gtbbox = Variable(gtbbox)
                if torch.cuda.is_available():
                    gtbbox = gtbbox.cuda()
                    img = img.cuda()
                imgs.append(img)
                gtbox.append({"boxes": gtbbox[:, 1:], "labels": gtbbox[:, 0]})
            opt.zero_grad()
            data = net(imgs, gtbox)
            l_cls = data['loss_classifier']
            l_box = data['loss_box_reg']
            l_obj = data['loss_objectness']
            l_box_rpn = data['loss_rpn_box_reg']
            bar('正在第%d轮训练,loss_cls=%.5f,loss_box=%.5f,loss_obj=%.5f,loss_box_rpn=%.5f' %
                (epoch, l_cls.data, l_box.data, l_obj.data, l_box_rpn.data), n, num_train)
            loss = l_cls + l_box + l_obj + l_box_rpn
            # # 添加监视目标
            # writer.add_scalar("loss_classifier", l_cls.data, (epoch - 1) * num_train + n)
            # writer.add_scalar("loss_box_reg", l_box.data, (epoch - 1) * num_train + n)
            # writer.add_scalar("loss_objectness", l_obj.data, (epoch - 1) * num_train + n)
            # writer.add_scalar("loss_rpn_box_reg", l_box_rpn.data, (epoch - 1) * num_train + n)
            loss.backward()
            opt.step()
        torch.save(net, cfg.FasterRCNN_checkpoints)


if __name__ == '__main__':
    train()
