"""
Transformer的输出部分：
    Linear线性层和Softmax激活函数
"""

import torch
import torch.nn as nn


class Output(nn.Module):
    def __init__(self, d_model, vocab_size):
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建神经网络结构
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, data):
        """
        训练阶段返回未经Softmax处理的原始分数logits。
        nn.CrossEntropyLoss内部已经包含LogSoftmax，因此不能在这里提前Softmax。
        :param data: 解码器最终的输出结果，形状[batch_size,seq_len,d_model]
        :return: logits，形状[batch_size,seq_len,vocab_size]
        """
        return self.linear(data)

    @staticmethod
    def probabilities(logits):
        """
        推理或展示时，将logits转换成板书中的Output Probabilities。
        每个位置在词汇表维度上的概率和为1。
        """
        return torch.softmax(logits, dim=-1)
