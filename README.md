## 概述
这是关于[Texture Synthesis Using Convolutional Neural Networks](https://arxiv.org/pdf/1505.07376v3.pdf)论文的tensorflow2.0代码实现，使用keras预训练的VGG19模型，依照论文重新更改了vgg19的设置。
本实现在论文给出的纹理中基本能得到复现结果

## 环境
tensorflow >2.0
numpy
Pillow
matplotlib

## Usage
``` python3 synthesize.py src.jpg --output output.jpg```

## 文件
- custome_vgg.py 
 对vgg19重做修改
- utils.py
文件读写，格拉姆矩阵计算等
- Texture Synthesis Using Convolutional Neural Networks.ipynb 关于代码实现的实验代码和思路讲解

## result


![原始图片](https://upload-images.jianshu.io/upload_images/19490456-6aa5f454e242afd6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![合成结果](https://upload-images.jianshu.io/upload_images/19490456-1b2d9855badabfaf.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![原始图片](https://upload-images.jianshu.io/upload_images/19490456-0b9de3091ce741cc.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![合成结果](https://upload-images.jianshu.io/upload_images/19490456-0402f05670b53e25.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![原始图片](https://upload-images.jianshu.io/upload_images/19490456-bdd2e4820ee9edd0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![合成结果](https://upload-images.jianshu.io/upload_images/19490456-c79dd4d7ba70e163.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 讨论
- 对于合成纹理中的高频噪声可以加入总变分损失(Total variation loss)优化，但笔者由于调参原因找不到合适的参数，因此放弃
- 同时，使用直方图匹配在深色图片上也会起到优化作用，但泛化性不强