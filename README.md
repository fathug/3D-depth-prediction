# 3D-depth-prediction
3D depth prediction based on neural network and coded Light

基于语义分割模型（本repository使用的是改进的U-Net模型），实现单帧条纹图到深度图的预测  
Single-frame fringe map to depth map prediction based on the semantic segmentation model (this repository uses  U-Net model as baseline)  

### Prerequisites：
```
PyTorch 1.8.x+cu102
NVIDIA CUDA 11.6
```

### How to implement model training:  
1. Preprocess pics data (512pixel or others) and label data.  
path is './Dataset/image' and './Dataset/seg_image'  
2. Customize the input and output of the network layer.  
Run train.py and predict.py by turn.

### author read:  
file copy from: PythonSpace/labelme_test  
![20220905165348](https://cdn.jsdelivr.net/gh/fathug/PicBed@main/picgo/20220905165348.png)  
refer: 多阶段深度学习单帧条纹投影三维测量方法