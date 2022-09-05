# -*- coding:utf-8 -*-
# @Time : 2022/5/24 18:03
# @Author : 
# @Note : 1.使用新的训练标签（经过24转8位的灰度标签图），地址为Dataset4


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
import time

from utils import *
from data import *
from net import *


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')    # 显存不足,直接用CPU

# data_path = 'D:\Deeplearning\所有测试程序的项目_1217\Unet_0404\VOCdevkit\VOC2007'       # 加 r 转义
data_path = './Dataset4'     # 训练集路径
weight_path = './param/unet.pth'    # 权重的保存路径
dis_image_path = './dis_image4'      # 训练结果图路径


if __name__ == '__main__':
    mydataset = MyDataset(data_path)        # 实例化dataset
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=False)   # bs小一点,否则显存不足
    # note:如果显卡性能不足，系统报错。解决办法：调低batch。

    unet = UNet()
    unet = unet.to(device)

    # 增加模型权重的载入--------------------------------
    if os.path.exists(weight_path):
        unet.load_state_dict(torch.load(weight_path))
        print('weight load successfully!')
    else:
        print('no weight file!')
    # ------------------------------------------------

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(unet.parameters())

    # 打印当前时间
    print('开始训练:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # epoch = 1
    for epoch in range(2):
        for i, (image, seg_image) in enumerate(mydataloader):
            image, seg_image = image.to(device), seg_image.to(device)
            out_image = unet(image)

            train_loss = loss_func(out_image, seg_image)
            optimizer.zero_grad()   # 清空梯度
            train_loss.backward()   # 反向计算
            optimizer.step()    # 更新梯度

            if i % 2 == 0:
                print(f'epoch:{epoch}, step:{i}, train_loss:{train_loss.item()}')      # 这里trainloss需要改一下，迭代值

            if epoch % 2 == 0:      # 每n个epoch保存权重1次
                torch.save(unet.state_dict(), weight_path)  # 保存网络权重

            _image = image[0]
            _seg_image = seg_image[0]
            _out_image = out_image[0]
            dis_image = torch.stack((_image, _seg_image, _out_image), dim=0)
            # save_image(dis_image, f'{dis_image_path}/{i}.jpg')
            save_image(dis_image, f'{dis_image_path}/{epoch}-{i}.jpg')      # 保存拼接过后的图片
        epoch += 1

    # 打印时间
    print('结束训练:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


