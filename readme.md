# Pytorch学习1 - 创建一个分类器

作为一个算法小白，AI工程小白，想要学习人工只能相关一些技术。
得利于当前各个人工智能算法框架的完善，像我这样的毫无数学基础，毫无算法基础的小白，
也能够参与进来，实现一些看上去很酷炫的功能。

经过一点点的调研，我决定通过Pytorch 这个人工智能框架来学习，深度神经网络的相关技术。
不求能够搞懂相关原理，只求，经过着一些列的学习之后，能够了解人工智能，神经网络的相关基础知识，
不能通过现有的、开源的框架实现一系列业务问题，最好能够实现一些业务调优。

这一篇笔记，将根据Pytorch官网的60分钟入门教程，学习Pytorch 的基本使用，并且实现一个简单的分类器。

## 训练图像分类器
根据官网教程的说明，本次图像分类器教学，需要实现以下步骤：
1. 加载数据集，标准化数据，区分训练集与测试集
2. 定义CNN卷积神经网络
3. 定义损失函数
4. 在训练集上训练神经网络
5. 在测试集上测试神经网络

## 数据来源

Pytorch 的torchvision包，提供了一系列针对图像识别的训练集。包含Imagenet, CIFAR10, MNIST, 等等。
可直接下载使用，大大方便了神经网络的学习与开发。

## 加载数据集

```python
import torch
import torchvision
from torchvision.transforms import transforms

"""
定义图像处理方式：
通过Compose可以将多个图像变换给串联起来，
比如，这里现将PIL格式的图像转化为张量(tensor)，然后将数据正则化

torchvision.transforms中提供了许多内置的图像变换方法可供调用
更多可用的图像变换方法请参考 
https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-transform/
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""
定义训练集与测试集
这里使用的是CIFAR10数据集
该数据集包含10组32x32的三通道图片
包含：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’
数据集将会下载
"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

"""
训练或测试时分批加载数据集，
减少运算负担
有时也可以根据批次计算梯度，快速学习

这里没有定义采样器切分训练集和测试集，而是使用了所有数据
如果需要，可以使用shuffle 配合sampler将数据集加以区分
torch.utils.data.sampler.SubsetRandomSampler()
使用从未参与训练的数据验证模型，达到验证模型的效果
"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

## 定义卷积神经网络 

###神经网络架构
这里为了搞懂示例代码中的各个参数的含义，大概学习了一下这个分类器的神经网络架构。
1. 针对输入图像，第一层卷积
- 输入为32x32像素的RGB图像，可表示为，32x32x3的矩阵
- 针对每幅图像提取6种特征，也即6个卷积核，每个卷积核可以认为是一个特征提取器，或者滤波器
- 示例中使用5x5的卷积窗口对图像进行卷积
- 使用卷积核与卷积窗口进行內积运算，內积越大，匹配程度越高
- 因此，卷积后输出将成为 28x28x6的矩阵
- 其中 28 = 32 - 5 + 1，窗口在行货列能够滑动的次数，6就是输出通道数，也即卷积核个数
2. 针对卷积结果，池化
- 池化的作用在于保持特征不变的同时，减少输入参数（降维），以防止过拟合
- 常用的最大池化，可以认为就是在压缩、模糊化图像，排除部分细节，使特征更为明显
- 示例中，使用2x2的池化窗口，也即，在2x2的窗口中取最大值
- 池化后，得到的结果被压缩1/4, 得到14x14x6的矩阵
3. 针对池化结果，做第二层卷积
- 与步骤一的卷积并无本质不同，只是输入变为池化结果
- 这一层可以加大特征提取数量，获得更为复杂的特征
- 示例中，将输出通道数设置为16，使用5x5的窗口滑动，得到 10x10x16 的矩阵
4. 再次池化
- 与步骤2完全区别，只是出入变成了第二次池化的结果
- 最后能够得到 5x5x16的矩阵
5. 将所有特征，拉成一维
6. 全连接前馈
7. 使用softmax进行分类，输出最终结果
8. 前馈方法
- 使用relu作为激活函数，使得分类器得到非线性分类的能力
- 激活函数的作用是能够给神经网络加入一些非线性因素，使得神经网络可以更好地解决较为复杂的问题。
- relu函数是一条以0位分界线的一条射线
- relu能在SGD中快速收敛，常常与SGD(随机梯度下降)配合使用
- relu可能存在神经元死亡问题，一旦梯度为零，则之后永远为零，权重将不再更新

一些小结：
- 通常情况下，靠近输入的卷积层，譬如第一层卷积层，会找出一些共性的特征
- 越往后，卷积核设定的数目越多，越能体现label的特征就越细致，就越容易分类出来
- 并且，改变卷积核的大小与卷积核的数目会对结果产生一定影响
- 缩小卷积核尺寸，增加卷积核数目都会提高准确率，但会大大增加计算量

### 代码

```python

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        """
        定义卷积神经网络

        这里继承torch.nn包中的默认Module，
        并将默认的输入从灰度图改为RGB三通道图
        """
        super(Net, self).__init__()
        # 第一个卷积层
        # 参数1：输入通道3，输入为RGB图像，为三通道图像
        # 参数1：输出通道数，即卷积核个数6，也即，需要提取的特征数
        # 参数1：卷积核窗口大小5，即5x5窗口，也即，匹配模板的窗口大小
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化，可以理解为模糊化，池化窗口为2
        # 在2x2的窗口中，取內积最大的值
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层（第一层的输出为第二层的输入通道数6，输出通道16，核心数5）
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层（输入单元数，输出单元数）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 损失函数，优化器
分类器可能在某些目标或者某些场景下表现良好，但对某些目标识别率极差，这就说明，此处的权重选择并不合理。

此外，我们还可以设置一个函数定量地衡量任意一个W的好坏，将W作为输入，得分作为输出，这个函数即为损失函数。

而要找到一种有效的方式来从W的可行域中找出结果表现最佳的W取值，这个过程则被成为优化过程。

```python
import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## 训练模型
```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

## 运行结果

```python
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter_test = iter(testloader)
    images, labels = dataiter_test.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

```
# 参考
[Pytorch官网入门教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
https://www.imooc.com/article/35821
