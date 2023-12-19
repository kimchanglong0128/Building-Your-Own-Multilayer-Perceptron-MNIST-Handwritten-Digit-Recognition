""" ReLU激活层 """

import numpy as np

class ReLULayer():
    def __init__(self):
        self.trainable = False

    def forward(self, Input):
        # 对输入应用ReLU激活函数并返回结果
        self.input = Input  # 设置input属性
        output = np.maximum(0, Input)
        return output

    def backward(self, delta):
        # 根据delta计算梯度
        # ReLU激活函数的导数是在输入大于0时为1，否则为0
        gradient = delta * (self.input > 0)
        return gradient
