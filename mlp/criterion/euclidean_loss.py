""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.

    def forward(self, logit, gt):
        """
        输入: (minibatch)
        - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
        - gt: 真实标签, 尺寸(batch_size, 10)
        """
        self.logit = logit  # 设置logit属性
        self.gt = gt   # 计算平均准确率和损失
        batch_size = logit.shape[0]
        diff = logit - gt
        self.loss = np.sum(diff ** 2) / (2 * batch_size)
        self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / batch_size

        return self.loss

    def backward(self):
        # 计算并返回梯度，梯度是logit和gt之间的差
        gradient = (self.logit - self.gt) / self.logit.shape[0]
        return gradient
