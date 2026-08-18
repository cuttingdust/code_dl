from aiohttp._websocket import mask

from demo01_input import *

import torch
import torch.nn as nn
from torch import Tensor

import math
import copy


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


def clones(model_obj, nums):
    """
    创建指定个数的相同网络结构对象
    :param model_obj: 网络结构对象
    :param nums: 个数
    :return: 网络结构对象列表
    """
    # 语法 nn.ModuleList([copy.deepcopy(模型对象) for _ in range(深拷贝复制的个数)])
    return nn.ModuleList([copy.deepcopy(model_obj) for _ in range(nums)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        """
        初始化
        :param d_model: 词向量维度/隐藏层隐藏状态向量维度。例如：512
        :param head: 多头的头数。例如：8
        :param dropout_p: 随机失活概率
        """

        # 1- 确定d_model能够被head整除
        assert d_model % head == 0

        # 2- 初始化父类
        super().__init__()

        # 3- 设置属性值
        self.d_model = d_model
        self.head = head
        self.head_dim = d_model // head  # 每个头分别处理的数据维度。例如：64
        self.dropout = nn.Dropout(p=dropout)

        # 4- 搭建网络结构
        """
            4个线性层的作用如下：
                第1个线性层：专门用来query进行线性处理，决定“当前词在找什么东西”
                第2个线性层：专门用来key进行线性处理，决定“其他词能够提供什么信息”
                第3个线性层：专门用来value进行线性处理，决定“其他词的信息的重要程度是什么样的”
                第4个线性层：对多头并行处理，并且concat拼接后的张量做最终的线性处理，让数据变得更加平稳
        """
        self.linear_list = clones(
            nn.Linear(in_features=d_model, out_features=d_model), 4
        )

        # 5- 权重张量
        self.weight = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        """
        前向传播：多头注意力计算
        :param query: 查询张量，形状：[batch_size每个批次中有多少条句子,seq_len每条句子中有几个词,d_model词向量的维度/隐藏状态维度]
        :param key: 键张量，形状：[batch_size,seq_len,d_model]
        :param value: 值张量，形状：[batch_size,seq_len,d_model]
        :param mask: 掩码张量，形状：[head,seq_len,seq_len]。注意第一个维度代表的是头数
        :return:
        """
        # 1- 掩码处理：进行升维，3维变4维
        if mask is not None:
            # 例如：[8,4,4] -> [1,8,4,4]
            # 不管每个批次中有多少条句子，每条句子用的掩码是同一份
            mask = mask.unsqueeze(0)

        # 2- 获得batch_size，也就是批次中句子的条数
        batch_size = query.shape[0]

        # 3- 前3个线性层分别并行对QKV进行处理
        # 方式一：分开版
        # [(Linear,query), (Linear,key), (Linear,value)]
        linear_output_list = []
        model_and_data_list = list(zip(self.linear_list, (query, key, value)))
        for model, data in model_and_data_list:
            """
            1- model(data)：线性变换
            2- reshape：[2,4,512]->[2,4,8,64]
            3- transpose：[2,4,8,64]->[2,8,4,64]
            """
            model_output = model(data)
            reshape_output = model_output.reshape(
                batch_size, -1, self.head, self.head_dim
            )
            linear_output_list.append(reshape_output.transpose(1, 2))
        new_query, new_key, new_value = linear_output_list

        # 方式二：合并版【理解】
        # new_query, new_key, new_value = [
        #     model(data)
        #     .reshape(batch_size, -1, self.head, self.head_dim)
        #     .transpose(1, 2)
        #     for model, data in list(zip(self.linear_list, (query, key, value)))
        # ]

        # 4- 多头并行计算注意力
        C, weight = attention(
            new_query, new_key, new_value, mask=mask, dropout=self.dropout
        )
        self.weight = weight

        # 5- 对多头处理后的数据进行拼接
        # [2,8,4,64] -> [2,4,8,64] -> [2,4,512]
        result = C.transpose(1, 2).reshape(batch_size, -1, self.head * self.head_dim)

        # 6- 调用最后一个线性层对拼接后的数据进行处理，让模型更加稳定
        return self.linear_list[-1](result)


# 测试多头注意力计算
def use_multi_head_attention():
    # 1- 获取位置编码之后的词数据
    position_data = use_positional_encoding()

    # 2- query、key、value参数
    query = key = value = position_data

    # 3- 创建多头注意力实例对象
    mask = torch.triu(torch.ones(size=(8, 4, 4)))
    my_attention = MultiHeadAttention(d_model=512, head=8, dropout=0.1)

    # 4- 调用前向传播
    result = my_attention(query, key, value, mask=mask)
    print(f"多头注意力计算结果：{result.shape}")
    print(f"多头注意力计算结果：{result}")

    return result


if __name__ == "__main__":
    # use_attention()

    use_multi_head_attention()
