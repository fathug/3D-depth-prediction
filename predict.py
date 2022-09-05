# -*- coding:utf-8 -*-
# @Time : 2022/4/7
# @Author : 
# @Note :
import os
import torch
from net import UNet
from utils import *
from data import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt



net = UNet()    # 实例化操作,此时网络没有权重(所以需要下面载入权重)

weight_path = './param/unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('load successfully!')
else:
    print('no net weight!')

# 网络设置完成,下面处理需要喂入的数据
# 需要对图片做的操作:(W,H)的大小, 升维度,
print('input path of unpredict image:')
unpred_img = input()

img = ResizeImage(unpred_img)
img = transform(img)
# print(img.shape)    # 这里打印出来图片shape为 [3, 256, 256]

img = img.unsqueeze(0)  # 升维后才能喂入网络

img_out = net(img)  # 预测结果
save_image(img_out, './result/result4-1.jpg')


img_out = img_out.squeeze(0)    # 去掉第一个维度
print(img_out.shape)        # 预测图片的维度
# plt.imshow()
# plt.show()