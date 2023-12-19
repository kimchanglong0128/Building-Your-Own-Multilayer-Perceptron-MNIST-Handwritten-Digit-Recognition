""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.loss = 0.
        self.acc = 0.

    def forward(self, logit, gt):
        """
        输入: (minibatch)
        - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
        - gt: 真实标签, 尺寸(batch_size, 10)
        """
        # 计算softmax并保存到self.softmax
        max_logit = np.max(logit, axis=1, keepdims=True)
        exp_logit = np.exp(logit - max_logit)
        softmax = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)

        self.softmax = softmax  # 设置softmax属性
        self.gt = gt
        batch_size = logit.shape[0]

        # 计算交叉熵损失
        self.loss = -np.sum(gt * np.log(softmax)) / batch_size
        
        # 计算准确率
        predictions = np.argmax(softmax, axis=1)
        ground_truth = np.argmax(gt, axis=1)
        correct_predictions = np.sum(predictions == ground_truth)
        self.acc = correct_predictions / batch_size

        return self.loss

    def backward(self):
        # 计算并返回梯度
        gradient = (self.softmax - self.gt) / self.softmax.shape[0]
        return gradient
