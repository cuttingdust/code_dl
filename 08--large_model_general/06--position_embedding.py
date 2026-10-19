#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2025/12/8 16:51
@File    : _06_position_embedding.py
@Function :
演示没有位置编码时，自注意力具有“置换等变性”：

1. 如果只是重新排列输入token的顺序，输出也只会按照相同顺序重新排列；
2. 同一个token从一个位置移动到另一个位置后，对齐该token进行比较，
   它的注意力输出基本不变；
3. 这并不是说所有位置的输出都相同，而是说明纯自注意力无法仅凭自身
   判断token原来位于第几个位置，因此Transformer还需要额外的位置编码。
"""

import torch
import torch.nn as nn

# 1- 定义参数
d_model = 8  # 每个token使用8维向量表示，也是多头注意力的输入、输出维度
num_heads = 2  # 注意力头数量；每个头处理的维度为8 / 2 = 4
seq_len = 4  # 一条句子包含4个token
batch = 1  # 一个批次中只有1条句子

print()

# 2- 创建多头注意力层（Multi-Head Attention，简称MHA）
mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    # 设置后，输入张量形状为[batch_size, seq_len, d_model]，
    # 本例就是[1, 4, 8]。
    batch_first=True,
)


# 3- 创建不包含位置编码的输入
# x的形状是[1, 4, 8]：1条句子、4个token、每个token是8维向量。
# 可以把4个token暂时想象成[A, B, C, D]。
# torch.randn只是用随机向量模拟真实模型中的token embedding。
x = torch.randn(batch, seq_len, d_model)

# 4- 对原始输入执行多头自注意力
# query、key、value都是x，因此这里计算的是“自注意力”。
# y1形状为[1, 4, 8]，包含4个token经过注意力计算后的结果；
# attn1形状为[1, 4, 4]，表示4个token之间的注意力权重。
y1, attn1 = mha(x, x, x)

# 5- 重新排列token的顺序
# 原始索引：[0, 1, 2, 3]，即[A, B, C, D]
# 新的索引：[2, 0, 3, 1]，即[C, A, D, B]
perm = torch.tensor([2, 0, 3, 1])

# 这里只改变第1维（seq_len维度）的token顺序，没有打乱最后一维的8个特征。
# 因此这是“移动完整token”，不是“交换token内部的特征维度”。
x_perm = x[:, perm, :]

print("原始输入: ", x)
print("置换后的输入: ", x_perm)

print("===" * 50)
print("原始输出: ", y1)

# 6- 对重新排列后的输入再次执行同一个多头自注意力层
# 没有位置编码时，自注意力不知道C现在位于第1个位置，
# 它只知道A、B、C、D这些向量之间的内容关系。
# 因此输出将从[YA, YB, YC, YD]变成[YC, YA, YD, YB]。
y2, attn2 = mha(x_perm, x_perm, x_perm)
print("置换后的输出: ", y2)

# 7- 为了比较“同一个token”换位置前后的输出，需要先把y1按相同顺序重排。
# 不能直接比较y1与y2，因为此时相同数组下标上放的是不同token，例如：
# y1[:, 0, :]属于A，而y2[:, 0, :]属于C。
#
# 将y1变成[YC, YA, YD, YB]后，才是在比较：
# 原来C的输出 与 移动到第1位后的C的输出、
# 原来A的输出 与 移动到第2位后的A的输出……
y1_perm = y1[:, perm, :]

# 8- 理论上满足y2 == y1_perm。
# torch.allclose用于判断两个浮点张量是否在允许误差内接近；
# atol=1e-5表示允许存在很小的浮点计算误差。
# 输出True说明：没有位置编码时，改变token顺序只会使输出跟着重新排列，
# 自注意力本身没有获得“第1位、第2位……”这样的位置信息。
print("是否满足置换等变性：", torch.allclose(y1_perm, y2, atol=1e-5))
