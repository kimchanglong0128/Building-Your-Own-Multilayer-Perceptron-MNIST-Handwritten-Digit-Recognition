""" 全连接层 """

import numpy as np

class FCLayer():
    def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
        """
        对输入进行线性变换: y = Wx + b
        参数简介:
            num_input: 输入大小
            num_output: 输出大小
            actFunction: 激活函数类型(无需修改)
            trainable: 是否具有可训练的参数
        """
        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.actFunction = actFunction
        assert actFunction in ['relu', 'sigmoid']

        self.XavierInit()

        self.grad_W = np.zeros((num_input, num_output))
        self.grad_b = np.zeros((1, num_output))

        self.input = None  # 储存向前传播的输入

    def forward(self, Input):
        # 保存前向传播的输入，以备反向传播使用
        self.input = Input

        # 计算线性变换
        linear_output = np.dot(Input, self.W) + self.b

        # 应用激活函数
        if self.actFunction == 'relu':
            output = np.maximum(0, linear_output)  # ReLU激活函数
        elif self.actFunction == 'sigmoid':
            output = 1 / (1 + np.exp(-linear_output))  # Sigmoid激活函数

        return output

    def backward(self, delta):
        # 计算权重self.W的梯度
        self.grad_W = np.dot(self.input.T, delta)

        # 计算偏置self.b的梯度
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        # 计算反向传播的delta，根据激活函数类型
        if self.actFunction == 'relu':
            delta_next = np.dot(delta, self.W.T) * (self.input > 0)
        elif self.actFunction == 'sigmoid':
            delta_next = np.dot(delta, self.W.T) * self.input * (1 - self.input)

        return delta_next

    def XavierInit(self):
        # 初始化，无需了解.
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        if 'relu' == self.actFunction:
            init_std = raw_std * (2**0.5)
        elif 'sigmoid' == self.actFunction:
            init_std = raw_std
        else:
            init_std = raw_std # * 4

        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
