import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fix BrokenPipeError on Windwos

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
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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


net = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    dataiter_test = iter(testloader)
    images, labels = dataiter_test.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
