# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午8:51
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : Config.py

class Config():
    min_size = 600
    anchor_size = [(128, 128), (90, 180), (180, 90), (256, 256), (180, 360), (360, 180), (512, 512),
                   (360, 720), (720, 360)]
    scale = 32 #特征图层相比与原图的压缩尺寸，MobileNet是32倍
    feature_map_channels = 1280 #特征层的通道数