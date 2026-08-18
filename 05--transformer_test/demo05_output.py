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
        :param data: 解码器最终的输出结果，形状[batch_size,seq_len,d_model]
        :return:
        """

        return torch.softmax(self.linear(data), dim=-1)
