""" SGD优化器 """

import numpy as np

class SGD():
    def __init__(self, learningRate, weightDecay):
        self.learningRate = learningRate
        self.weightDecay = weightDecay

    # 一步反向传播，逐层更新参数
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                # 计算参数的梯度
                gradient_W = layer.grad_W
                gradient_b = layer.grad_b

                # 计算权重的更新差异
                diff_W = gradient_W + self.weightDecay * layer.W
                diff_b = gradient_b

                # 更新权重和偏置
                layer.W -= self.learningRate * diff_W
                layer.b -= self.learningRate * diff_b
