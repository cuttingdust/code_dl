"""
Transformer的输入部分：包含如下内容
    词嵌入层：输入词索引，得到词向量
    位置编码
"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.vocab_size = vocab_size  # 词汇表大小
        self.d_model = d_model  # 词向量的维度。例如：512

        # 3- 搭建网络结构：只有一个词嵌入层
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, input):
        """
        前向传播。输入一条句子，得到词向量
        :param input: 一条句子，里面的元素是词索引。张量形状[batch_size每个批次中句子的条数,seq_len每条句子中词的个数]
        :return: 词向量
        """

        """
            为什么要乘以math.sqrt(self.d_model)，也就是根号dk？
            答：为了对数据进行缩放，避免与位置编码的数据值之间的大小差异过大。为了让模型训练稳定，也就是缓解梯度消失或梯度爆炸
                词向量维度越大，越容易出现极小值
        """
        return self.embed(input) * math.sqrt(self.d_model)


def use_embedding():
    # 1- 创建词嵌入层类的实例对象
    # my_embed = Embedding(vocab_size=1000, d_model=300)
    my_embed = Embedding(vocab_size=1000, d_model=10240)

    # 2- 准备数据
    # 注意：目前情况下，词索引的取值区间[0,999]
    x = torch.tensor(
        [
            # 单词索引
            [100, 2, 666],
            [500, 888, 421],
        ]
    )

    # 3- 输入句子，得到词向量
    word_vector = my_embed(x)
    print(f"词向量的形状{word_vector.shape}")  # 2条句子，3句子中3个词，5词向量维度
    print(f"词向量的数据{word_vector}")
    print(
        f"词向量的数据{word_vector.abs().min()}"
    )  # 获得乘以根号dk以后数据中绝对值的最小值


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=60):
        """
        位置编码初始化方法
        :param d_model: 词向量维度。例如：512
        :param dropout: 神经元随机失活概率
        :param max_len: 能够处理的句子最大长度。也就是词的个数
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 创建随机失活层
        self.dropout = nn.Dropout(p=dropout)

        # 3- 定义pe(也就是位置编码张量)
        pe = torch.zeros(size=(max_len, d_model))  # 目前的形状[60,d_model]

        # 4- 定义列向量，用来存储输入句子中词索引信息
        position = torch.arange(0, max_len).unsqueeze(1)  # 形状[60,1]

        # 5- 得到pos/分母公式中的，1/分母
        div_term = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

        # 6- 计算pos/分母的结果
        position_value = position * div_term

        # 7- 调用sin、cos分别计算位置编码值
        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)

        # 8- 调整pe的形状变成3维，也就是[60,d_model]->[1,60,d_model]每个批次1条句子，每个句子最多60个词，词向量是d_model
        pe = pe.unsqueeze(0)

        # 9- 将pe的值注册到缓存中，通过它来不断的更新位置编码信息。后面可以直接通过self.pe调用
        self.register_buffer("pe", pe)

    def forward(self, embed):
        """
        位置编码的前向传播：多条句子中所有词的词向量，和位置编码张量进行求和操作，返回结果
        :param embed: 词向量
        :return:
        """

        """
            embed.shape[1]：embed的形状[batch_size,seq_len,d_model]，得到多少个词
            为什么self.pe[:, :embed.shape[1]]？
            如果这条句子中词的个数超过了max_len，那么最多只对前max_len个词的位置编码进行相加
        """
        result = embed + self.pe[:, : embed.shape[1]]
        # print(f"对应的位置编码值：{self.pe[:, :embed.shape[1]]}")
        return self.dropout(result)


def use_positional_encoding():
    d_model = 512

    # 实例化词嵌入层类的对象
    my_embed = Embedding(vocab_size=1000, d_model=d_model)

    # 准备数据
    x = torch.tensor(
        [
            # 单词索引
            [100, 2, 421, 600],
            [500, 888, 421, 615],
        ]
    )

    # 输入数据，得到对应的词向量
    word_embed = my_embed(x)
    print(f"词向量的形状：{word_embed.shape}")
    print(f"词向量的值：{word_embed}")

    # 创建位置编码
    my_pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=60)

    # 调用位置编码，最终效果是往词向量中加上了位置编码的值
    result = my_pe(word_embed)
    print(f"最终的形状：{result.shape}")
    print(f"最终的值：{result}")


# 可视化位置编码
def plot_position():
    # 1. 实例化位置编码器.
    my_position = PositionalEncoding(d_model=20, dropout=0.1, max_len=100)

    # 2. 生成全0的输入, 观察位置编码的模式.
    # (1, 100, 20) -> 批次大小, 句子长度, 词嵌入维度
    embed = torch.zeros(1, 100, 20)
    y = my_position(embed)

    # 3. 设置图表大小.
    plt.figure(figsize=(20, 15))
    # 绘制位置编码第4到第7列, 100个词的  [4, 5, 6, 7]
    """
        图形的信息解释：
            x轴：词的索引，目前总共有100个词
            y轴：位置编码值
    """
    plt.plot(np.arange(100), y[0, :, 4:8].detach().numpy())
    plt.legend([f"dim {p}" for p in [4, 5, 6, 7]])
    plt.show()


if __name__ == "__main__":
    # use_embedding()

    # use_positional_encoding()

    plot_position()
