#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/18 12:27
@File    : _07_scaled_attention.py
@Function :
通过对比“未缩放点积注意力”和“缩放点积注意力”，演示Transformer为什么使用：

    Softmax(Q @ K.T / sqrt(dk))

当每个注意力头的向量维度dk越来越大时，Q与K的点积数值范围也会扩大，
这会使Softmax过早接近0或1，产生饱和现象。除以sqrt(dk)后，可以把
注意力分数控制在相对稳定的范围，使概率分布和反向传播梯度更加稳定。
"""

import math

import torch
import torch.nn.functional as F

# 固定随机种子，使每次运行生成相同的Q、K，方便对照实验结果。
torch.manual_seed(0)


def attention_test(dk, seq_len=10):
    """
    测试不同dk下，是否使用sqrt(dk)缩放对注意力分数和Softmax的影响。

    :param dk: 一个注意力头中Query和Key向量的维度。
               例如d_model=512、num_heads=8时，通常dk=512/8=64。
    :param seq_len: 句子中的token数量，默认一条句子包含10个token。
    """
    print(f"\n===== dk = {dk} =====")

    # 1- 构造Query和Key
    # Q、K的形状都是[seq_len, dk]。
    # 例如seq_len=10、dk=64时，形状为[10, 64]，表示：
    #   1- 一共有10个token；
    #   2- 每个token分别使用一个64维Query向量和64维Key向量表示。
    #
    # requires_grad=True表示PyTorch会为Q记录计算过程，之后可以反向传播梯度。
    # 当前示例只观察logits和Softmax，没有调用backward()，所以不会实际打印梯度。
    Q = torch.randn(seq_len, dk, requires_grad=True)
    K = torch.randn(seq_len, dk)

    # 2- 计算未缩放的注意力分数
    # “@”不是叉乘，而是矩阵乘法，等价于：
    #
    #     torch.matmul(Q, K.T)
    #
    # K.T是将二维矩阵K进行转置：
    #
    #     Q.shape   = [seq_len, dk]      例如[10, 64]
    #     K.shape   = [seq_len, dk]      例如[10, 64]
    #     K.T.shape = [dk, seq_len]      例如[64, 10]
    #
    # 因此矩阵乘法的形状变化是：
    #
    #     [10, 64] @ [64, 10] -> [10, 10]
    #
    # 中间两个64必须相同，结果保留外侧的两个10。
    # 输出矩阵中的每一个元素，都是一个Query向量和一个Key向量的点积：
    #
    #     logits_no_scale[i, j] = Q[i] · K[j]
    #
    # 例如Q[i]和K[j]都是3维向量时，点积计算为：
    #
    #     q1*k1 + q2*k2 + q3*k3
    #
    # 所以“整体操作”是矩阵乘法，而结果中的每一个分数来自向量点积。
    # 叉乘通常只用于三维向量，会返回一个新的三维向量；这里完全不是叉乘。
    logits_no_scale = Q @ K.T

    # dim=-1表示对注意力分数矩阵的最后一个维度执行Softmax。
    # 对每个Query来说，它对全部Key的注意力概率之和等于1。
    attn_no_scale = F.softmax(logits_no_scale, dim=-1)

    # 3- 计算缩放后的注意力分数
    # 一个点积会累加dk项随机数。dk越大，点积结果的波动范围通常越大：
    #
    #     点积结果的方差约为dk
    #     点积结果的标准差约为sqrt(dk)
    #
    # 因此除以sqrt(dk)，可以把不同dk下的注意力分数重新压回相近范围。
    # 这就是Transformer“缩放点积注意力”中的缩放操作。
    logits_scale = (Q @ K.T) / math.sqrt(dk)
    attn_scale = F.softmax(logits_scale, dim=-1)

    # 4- 打印缩放前后的logits范围
    # 如果没有缩放，随着dk增大，最小值和最大值通常会越来越极端；
    # 缩放以后，数值一般会保持在比较稳定的范围。
    print(
        "Logits (no scale): "
        f"min={logits_no_scale.min():.2f}, "
        f"max={logits_no_scale.max():.2f}"
    )

    print(
        "Logits (scaled):   "
        f"min={logits_scale.min():.2f}, "
        f"max={logits_scale.max():.2f}"
    )

    # 5- 打印Softmax中的最大概率，观察是否出现饱和
    # 如果最大概率非常接近1，就意味着其他概率基本接近0。
    # 这种极端概率如果只是由dk过大造成，而不是模型真正学到的结果，
    # 会使Softmax梯度接近0，导致训练变慢或不稳定。
    print("Softmax max prob (no scale): ", attn_no_scale.max().item())

    print("Softmax max prob (scaled):   ", attn_scale.max().item())


# 6- 分别测试不同的注意力头维度
# 预期现象：
#   1- dk越大，未缩放logits范围越大，Softmax最大概率逐渐接近1；
#   2- 使用sqrt(dk)缩放后，logits和最大概率保持相对稳定。
for dk in [8, 64, 256, 1024]:
    attention_test(dk)
