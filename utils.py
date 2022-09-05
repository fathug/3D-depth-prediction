# -*- coding:utf-8 -*-
# @Time : 2022/4/7
# @Author : 
# @Note : 对不规则的图片进行不变形缩放
from PIL import Image


def ResizeImage(path):
    image = Image.open(path)
    # print(image.size)
    max_size = (max(image.size), max(image.size))
    mask = Image.new('RGB', max_size, (0,0,0))
    mask.paste(image, (0,0))    # 粘贴到左上角
    mask = mask.resize((256, 256))
    return mask

# ResizeImage('VOCdevkit/VOC2007/JPEGImages/000005.jpg')
