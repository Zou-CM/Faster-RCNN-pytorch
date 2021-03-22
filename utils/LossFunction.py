# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 下午10:43
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : LossFunction.py

import torch

def smoothL1(a, b):
    if torch.abs(a - b) < 1:
        return 0.5 * (a-b) * (a-b)
    else:
        return torch.abs(a - b) - 0.5