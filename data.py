# -*- coding:utf-8 -*-
# @Time : 2022/4/7
# @Author : 
# @Note :

from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):   # path为 /VOCdevkit/VOC2007/
        self.path = path
        self.name = os.listdir(os.path.join(self.path, 'seg_image'))  # 遍历标签地址-》所有的变成名称

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]

        segment_path = os.path.join(self.path, 'seg_image/', segment_name)   # 标签地址
        image_path = os.path.join(self.path, 'image/', segment_name.replace('png', 'jpg'))        # 图片地址
        segment_resize = ResizeImage(segment_path)
        image_resize = ResizeImage(image_path)
        return transform(image_resize), transform(segment_resize)


# if __name__ == '__main__':
#     image = MyDataset('D:/Deeplearning/所有测试程序的项目_1217/Unet_0404/VOCdevkit/VOC2007/')
    # print(image[0])     # 索引为0的图片和标签
    # print(image[0][0])  # 索引为0的图片
    # print(image[0][1])  # 索引为0的标签




