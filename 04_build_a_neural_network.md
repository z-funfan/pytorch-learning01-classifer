# Pytorch学习4 - 神经网络一般构建步骤

根据前面的例子可以总结出我们构建神经网络的一般步骤，或者说是使用Pytorch构建神经网络的一般套路。

一般构建一个神经网络模型需要下面3大步骤：
1. 数据预处理：将输入数据经过一些列预处理，使得数据更适用于神经网络的计算，提高神经网络计算的运行效率
2. 模型构建：构建神经网络模型，通过循环训练调整各个节点的参数和权重，拟合真实数据
3. 模型验证：验证训练出的模型的准确率和效率，优化调整模型，防止过拟合

## 数据预处理
1. 类型变量、类型编码
- 输入数据中有很大一部分数据是非数值化的。比如，天气，晴、雨、多云等等，
或者男女等等类型变量，需要经过编码才能参与张量计算
- 常用的类型变量编码方法有，one-hot encoding，独热编码
- 主要是采用位状态寄存器来对个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效
- 比如，一周从周一到周日，可以分别编码为，1000000, 0100000, 0010000, 0001000, 0000100, 0000010, 0000001
2. 数据标准化
- 对于不同类型的数据，他们的取值范围可能存在很大差异。比如，温度，湿度，出生年月等等
- 为了方便后续计算，一般预处理时，假设数据符合正态分布，将数据正则化至(-1, 1)的取值范围
- 标准化数据需要保证数据分布的等效性
3. 数据切分
- 将所有数据切分为，训练集，测试集和验证集
- 训练模型时只是用训练集训练参数和权重
- 测试集和验证集用于测试验证训练后的模型的准确性，用于优化模型
- 一般训练时，可以将训练集分为多个批次进行训练，在每个去批次中不断迭代优化模型
- 比如分成128条数据每批次

## 模型构建
1. 构架模型
- 选取合适的模型网络
- Pytorch内置了许多成熟的网络模型，但是每个业务下是选取线性分类器，CNN，RNN等那种模型，需要经验和尝试
- 配置合适的隐含层数量以及节点数量
- 隐含层或节点单元数越多，就能够学习越复杂的特征；但是同时会大大增加计算的复杂度，并且容易出现过拟合的现象
2. 建立损失函数和优化器
- 损失函数，针对单个样本的误差，用于评估模型准确率，提供反馈信号
- 损失函数越小，代表拟合的越好
- 代价函数，损失函数的平均值，或者说是平均损失，用于评估总体的误差
- 优化器，用于优化网络参数，常用的有SGD，随机梯度下降优化
3. 训练模型
- 分批循环训练，根据反馈信号，优化参数和权重，直至损失函数达到稳定最小
## 模型验证
1. 损失函数下降曲线
2. 训练时间统计
3. 预测数据-实际数据对比