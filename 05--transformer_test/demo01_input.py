"""
Transformer的输入部分：包含如下内容
    词嵌入层：输入词索引，得到词向量
    位置编码
"""

import torch
import torch.nn as nn

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
    my_embed = Embedding(vocab_size=1000, d_model=300)
    # my_embed = Embedding(vocab_size=1000,d_model=10240)

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


if __name__ == "__main__":
    use_embedding()
