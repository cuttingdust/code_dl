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
        修改原因：原代码在这里直接Softmax；如果训练时再使用nn.CrossEntropyLoss，
        损失函数内部还会执行LogSoftmax，相当于重复处理，导致损失和梯度不符合预期。
        :param data: 解码器最终的输出结果，形状[batch_size,seq_len,d_model]
        :return: logits，形状[batch_size,seq_len,vocab_size]
        """
        return self.linear(data)

    @staticmethod
    def probabilities(logits):
        """
        推理或展示时，将logits转换成板书中的Output Probabilities。
        修改原因：把Softmax单独放在此方法中，可以同时满足“训练使用logits”和
        “预测时查看概率”两种需求；每个位置在词汇表维度上的概率和为1。
        """
        return torch.softmax(logits, dim=-1)
