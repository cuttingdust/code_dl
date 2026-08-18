from aiohttp._websocket import mask

from demo01_input import *

import torch
import torch.nn as nn
from torch import Tensor

import math


def attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None, dropout=None
):
    """
    注意力计算方法。能够涵盖如下几种情况：
        1- 编码器端的自注意力：query=key=value，mask为None
        2- 解码器端的掩码自注意力：query=key=value，mask不为None
        3- 解码器端的交叉注意力：query和key、value不等，但是key和value相同，mask为None
    :param query: 查询张量，形状：[batch_size每个批次中有多少条句子,seq_len每条句子中有几个词,d_model词向量的维度/隐藏状态维度]
    :param key: 键张量，形状：[batch_size,seq_len,d_model]
    :param value: 值张量，形状：[batch_size,seq_len,d_model]
    :param dropout: 随机失活Dropout层对象
    :param mask: 掩码张量，形状：[batch_size,seq_len,seq_len]
    :return: 专属信息包,权重张量。类型是元组
    """

    # 1- 获得d_k：词向量的维度
    d_k = query.shape[-1]

    # 2- Q和K的转置相乘；再除以根号d_k。得到相似性得分scores
    # K的转置：由  [batch_size,seq_len,d_model]  变成了  [batch_size,d_model,seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 3- 可选：对相似性得分scores进行掩码处理
    if mask is not None:
        # 对需要进行掩码的地方，将值替换成-1e9。注意：不要直接设置为0。需要和softmax结合理解。e的-1e9的结果趋近于0，表示权重趋近于0
        scores = scores.masked_fill(mask == 0, value=-1e9)

    # 4- 将相似性转成权重
    weight = torch.softmax(scores, dim=-1)

    # 5- 可选：随机时候，缓解过拟合
    if dropout is not None:
        weight = dropout(weight)

    # 6- 权重和value进行矩阵乘法，得到专属信息包
    C = torch.matmul(weight, value)

    return C, weight


def use_attention():
    # 1- 输入的数据先经过 词嵌入层 和 位置编码
    posi_embed = use_positional_encoding()

    # 2- 调用注意力计算方法
    # 2.1- 准备query、key、value参数
    query = key = value = posi_embed

    # 2.2- 准备掩码【可选】
    # (2,4,4)值的来源于use_positional_encoding的x。每个批次2条句子，每条句子4个词
    # 形状：[batch_size,seq_len,seq_len]
    mask = torch.triu(torch.ones(size=(2, 4, 4)))

    # 2.3- 随机失活网络层
    dropout = nn.Dropout(p=0.1)

    # 2.4- 调用
    # 2.4.1- 编码器端：没有掩码的自注意力
    encoder_attention_C, encoder_attention_weight = attention(
        query, key, value, dropout=dropout
    )
    print(f"编码器C：{encoder_attention_C.shape}-->{encoder_attention_C}")
    print(f"编码器C权重：{encoder_attention_weight.shape}-->{encoder_attention_weight}")

    # 2.4.2- 解码器端：有掩码的自注意力
    decoder_attention_C, decoder_attention_weight = attention(
        query, key, value, dropout=dropout, mask=mask
    )
    print(f"解码器C：{decoder_attention_C.shape}-->{decoder_attention_C}")
    print(f"解码器C权重：{decoder_attention_weight.shape}-->{decoder_attention_weight}")

if __name__ == "__main__":
    use_attention()
