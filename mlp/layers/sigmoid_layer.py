""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
    def __init__(self):
        """
        Sigmoid激活函数: f(x) = 1/(1+exp(-x))
        """
        self.trainable = False

    def forward(self, Input):
        # 对输入应用Sigmoid激活函数并返回结果
        self.input = Input  # 设置input属性

        output = 1 / (1 + np.exp(-Input))
        return output

    def backward(self, delta):
        # 根据delta计算梯度
        # Sigmoid激活函数的导数是 f(x) * (1 - f(x))
        gradient = delta * self.forward(self.input) * (1 - self.forward(self.input))
        return gradient
