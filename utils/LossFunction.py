# -*- coding: utf-8 -*-
# @Time    : 2021/3/22 下午10:43
# @Author  : Zou-CM
# @Email   : zou-cm@outlook.com
# @File    : LossFunction.py

import torch

def smoothL1(a, b):
    tmp = torch.abs(a - b)
    index_l = torch.lt(tmp, 1)
    index_b = torch.ge(tmp, 1)
    return torch.mean(0.5 * tmp * tmp * index_l + (tmp - 0.5) * index_b)
    # if torch.abs(a - b) < 1:
    #     return 0.5 * (a-b) * (a-b)
    # else:
    #     return torch.abs(a - b) - 0.5

def L2_loss(a, b):
    return torch.mean(torch.sqrt(a - b))