# 使用GPU加速新运算

之前谢了一个简单的图片分类器来区分图片中的内容。但是运行之后就发现，训练过程是在是太慢了，下面是默认使用CPU的运行结果。

```
Files already downloaded and verified
Files already downloaded and verified
[1,  2000] loss: 2.196
[1,  4000] loss: 1.882
[1,  6000] loss: 1.703
[1,  8000] loss: 1.588
[1, 10000] loss: 1.511
[1, 12000] loss: 1.452
[2,  2000] loss: 1.396
[2,  4000] loss: 1.347
[2,  6000] loss: 1.323
[2,  8000] loss: 1.341
[2, 10000] loss: 1.295
[2, 12000] loss: 1.261
Finished Training, cost 188.4 seconds
GroundTruth:    cat  ship  ship plane
```

训练大约12,000张图片（两轮）需要使用大约188.4秒，大概3分多钟，而且训练期间风扇狂转，CPU飙高，可能使我机器不行。
Pytoch是支持将张量放到GPU上并行计算的，因此只要是有NVIDIA或者AMD GPU的电脑并且支持CUDA就可以利用GPU加速计算。

关于如何安装CUDA可以参考之前写过的一篇笔记中的[环境安装](https://github.com/z-funfan/opencv-face-recognize#%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)

## 启用GPU加速
It’s very easy to use GPUs with PyTorch. You can put the model on a GPU:
在Pytorch中使用GPU很简单，首先将模型放入GPU
```python
device = torch.device("cuda:0")
model.to(device)
```
然后将张量也放入GPU计算
```python
mytensor = my_tensor.to(device)
```
**需要注意**的是，`my_tensor.to(device)`会返回一份张量的复制，并**不会修改原本**的张量，
因此把张量放入GPU时需要复制给一个新的变量。

默认情况下Pytorch仅使用一张GPU执行前馈和反馈操作，如果需要使用多张GPU并行操作，需要使用`DataParallel`方法
```python
model = nn.DataParallel(model)
```

## 修改原本代码
原先的模型默认运行在CPU上
```python
net = Net()
```

根据上面的教程，可将其运行在GPU上
```python
net = Net()
# 判断是否有可用的GPU，没有则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 判断是否存在多张GPU，如果存在，则使用DataParallel，并行计算
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)

net.to(device)
```

## 兼容问题

我的笔记本式2013年买的神州笔记本，使用的是NVIDIA GT645M独显，买的时候还算不错，但现在就有些不够看了。
运行的时候Pytorch给我跑了个错（感觉被鄙视了）

```
    Found GPU0 GeForce GT 645M which is of cuda capability 3.0.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability that we support is 3.5.
```

上网搜索了一下，说是使用安装包安装的Pytorch就会有这个问题，解决方法是从源码安装

### 从源码安装Pytorch
提示：期间肯能需要科学上网，最近正值祖国母亲的生日，科学上网遇到了一些问题。

首先，先要卸载原先从安装包安装的Pytorch
```
pip uninstall torch
```
然后，要安装一些编译环境，Windows7的话要安装VS2015，Windows10要安装VS2017以上，我本地安装的是VS2019.
确保CUDA环境已经安装，以及一些其他的依赖包
```
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```
下载源码并编译安装
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
set "VS150COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build"
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64
set DISTUTILS_USE_SDK=1
REM The following two lines are needed for Python 2.7, but the support for it is very experimental.
set MSSdk=1
set FORCE_PY27_BUILD=1
REM As for CUDA 8, VS2015 Update 3 is also required to build PyTorch. Use the following two lines.
set "PREBUILD_COMMAND=%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat"
set PREBUILD_COMMAND_ARGS=x64

call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
python setup.py install
```


验证
```
import torch
torch.cuda.is_available()

```