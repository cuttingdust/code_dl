from demo01_input import *

import torch
import torch.nn as nn
from torch import Tensor

import math
import copy


@MPoint(append_message="单层注意力")
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
    :param mask: 可广播到注意力分数的掩码张量；1/True表示允许关注，0/False表示屏蔽
    :return: 专属信息包,权重张量。类型是元组
    """

    # 1- 获得d_k：词向量的维度
    d_k = query.shape[-1]

    # 2- Q和K的转置相乘；再除以根号d_k。得到相似性得分scores
    # K的转置：由  [batch_size,seq_len,d_model]  变成了  [batch_size,d_model,seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 3- 可选：对相似性得分scores进行掩码处理
    if mask is not None:
        # 修改原因：mask可能是0/1浮点张量，也可能仍在CPU，而scores可能已经在GPU；
        # 不统一类型和设备会出现判断含义不清或CPU/CUDA设备不一致错误。
        # 因此统一转换成布尔掩码并移动到scores所在设备。
        # True表示允许关注，False表示屏蔽。使用当前数据类型的最小有限值，
        # 避免float16中直接填充-1e9发生溢出。
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, value=torch.finfo(scores.dtype).min)

    # 4- 将相似性转成权重
    weight = torch.softmax(scores, dim=-1)

    # 5- 可选：随机时候，缓解过拟合
    if dropout is not None:
        weight = dropout(weight)

    # 6- 权重和value进行矩阵乘法，得到专属信息包
    C = torch.matmul(weight, value)

    return C, weight


@MPoint(append_message="开始测试使用单层注意力")
def use_attention():
    # 1- 输入的数据先经过 词嵌入层 和 位置编码
    posi_embed = use_positional_encoding()

    # 2- 调用注意力计算方法
    # 2.1- 准备query、key、value参数
    query = key = value = posi_embed

    # 2.2- 准备掩码【可选】
    # (2,4,4)值的来源于use_positional_encoding的x。每个批次2条句子，每条句子4个词
    # 形状：[batch_size,seq_len,seq_len]
    # 修改原因：attention中使用mask==0的位置进行屏蔽，因此本项目约定1表示允许关注；
    # 在这个约定下，因果掩码必须使用下三角矩阵，才能允许看当前位置和过去、禁止看未来。
    mask = torch.tril(torch.ones(size=(2, 4, 4), dtype=torch.bool))

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


@MPoint(append_message="")
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
    def __init__(self, d_model, head, dropout_p=0.1):
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
        self.dropout = nn.Dropout(p=dropout_p)

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
        :param mask: 掩码张量，常见形状如下：
            [seq_len, seq_len]：所有批次、所有头共享一份因果掩码
            [batch_size, seq_len, seq_len]：每条句子使用独立掩码
            [batch_size, 1, query_len, key_len]：已经整理好的标准广播形状
        :return:
        """
        # 1- 将掩码整理成可以广播到[batch_size, head, query_len, key_len]的形状
        # 修改原因：原实现要求调用者为每个注意力头复制一份[head,seq,seq]掩码，
        # 不方便合并每条句子各自的PAD掩码；改成标准广播形状后，一份二维因果掩码
        # 就可以供所有批次和所有注意力头共享。
        if mask is not None:
            if mask.dim() == 2:
                # [query_len,key_len] -> [1,1,query_len,key_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # [batch_size,query_len,key_len] -> [batch_size,1,query_len,key_len]
                mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise ValueError(
                    f"mask应为2维、3维或4维张量，当前形状为{tuple(mask.shape)}"
                )

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
@MPoint(append_message="开始测试多头注意力")
def use_multi_head_attention():
    # 1- 获取位置编码之后的词数据
    position_data = use_positional_encoding()

    # 2- query、key、value参数
    query = key = value = position_data

    # 3- 创建多头注意力实例对象
    # 修改原因：所有头使用相同的因果规则，没有必要手工创建8份掩码；
    # 一份二维下三角因果掩码会自动广播到所有批次和所有注意力头。
    mask = torch.tril(torch.ones(size=(4, 4), dtype=torch.bool))
    my_attention = MultiHeadAttention(d_model=512, head=8, dropout_p=0.1)

    # 4- 调用前向传播
    result = my_attention(query, key, value, mask=mask)
    print(f"多头注意力计算结果：{result.shape}")
    print(f"多头注意力计算结果：{result}")

    return result


# 层归一化/规范化层：随着网络模型训练，数据有可能出现极值，导致梯度爆炸或消失，为了让模型训练稳定，通过标准化处理，让数据变的正常
# 也就是要实现y=kx+b，其中x要经过标准化处理（要计算数据的均值mean和标准差std）
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        # 1- 初始化父类
        super().__init__()

        # 2- 线性公式中参数的定义
        """
           为什么k和b要通过nn.Parameter进行定义？
           答：因为nn.Parameter会自动将k和b注册到神经网络模型中，作为可训练的参数。通过反向传播得到k和b。
               如果不写nn.Parameter，那么k和b的值永远固定，不会发生变化。
               对应前面神经网络模型代码中如下两行代码
               optimizer = optim.Adam(params=model.parameters())
               optimizer.step()
       """
        # 2.1- 定义k
        self.k = nn.Parameter(torch.ones(d_model))

        # 2.2- 定义b
        # 修改原因：标准LayerNorm初始化时缩放参数k为1、偏置b为0；
        # 原来把b初始化为1会让归一化结果整体向上平移1。
        self.b = nn.Parameter(torch.zeros(d_model))

        # 3- 定义小常数：防止分母为0
        self.eps = 1e-6

    def forward(self, data):
        """
        前向传播：输入前面子层处理后的数据，经过层归一化进行【标准化处理】，让模型训练稳定
        要实现y=kx+b，其中x要经过标准化处理（要计算数据的均值mean和标准差std）
        :param data: 前面子层处理后的数据。形状[batch_size,seq_len,d_model]
        :return:
        """
        # 1- 计算均值
        """
            参数解释：
                dim=-1：对最后一个维度计算均值
                keepdim=True：表示计算后，张量的形状保留。举例：[2,4,512]->[2,4,1]
        """
        mean = data.mean(dim=-1, keepdim=True)

        # 2- 计算总体方差。
        # 修改原因：LayerNorm统计的是当前词向量各维度的总体方差，应使用correction=0；
        # 原来的torch.std默认使用样本修正，在维度很小时甚至可能产生NaN。
        variance = data.var(dim=-1, keepdim=True, correction=0)

        # 3- 先标准化，再使用可训练参数k、b进行缩放和平移
        normalized_data = (data - mean) / torch.sqrt(variance + self.eps)
        return self.k * normalized_data + self.b


# 子层连接类
class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout_p):
        # 1- 初始化父类
        super().__init__()

        # 2- 创建层归一化
        """
           为什么把LayerNorm创建在__init__中？
           答：因为每个子层的后面都有该层
       """
        self.layer_norm = LayerNorm(d_model)

        # 3- 创建随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, data, sublayer_obj):
        """
        前向传播：使用你指定的具体子层实例对象【sublayer_obj】对数据【data】进行指定的处理
        :param data: 输入到层中的数据
        :param sublayer_obj: 具体子层实例对象。例如：多头自注意力实例对象、前馈网络实例对象、掩码多头自注意力实例对象、多头注意力实例对象（交叉注意力）
        :return:
        """

        # 具体子层实例对象【sublayer_obj】对数据【data】进行指定的处理
        # 写法一：原始Transformer的实现
        # 子层处理(也就是调用forward) -> 随机失活层 -> 残差连接 -> 层归一化
        # result = self.layer_norm(self.dropout(sublayer_obj(data)) + data)

        # 写法二：Pre-LN结构。先统一数值尺度再进入子层，有助于深层网络稳定传播梯度。
        # 注意：LayerNorm只规范均值和方差，并不保证数据变成高斯分布。
        # 层归一化 -> 子层处理(也就是调用forward) -> 随机失活层 -> 残差连接
        """
            第一步：层归一化        self.layer_norm(data)
            第二步：子层处理        sublayer_obj(self.layer_norm(data))
            第三步：随机失活层       self.dropout(sublayer_obj(self.layer_norm(data)))
            第四步：残差连接        + data
        """
        result = self.dropout(sublayer_obj(self.layer_norm(data))) + data

        return result


# 子层连接测试
@MPoint(append_message="测试子链接层")
def use_SubLayerConnection():
    # 1- 准备数据：词嵌入层+位置编码
    data = use_positional_encoding()
    d_model = 512

    # 2- 子层连接类的实例对象
    SubLayerConnection_obj = SubLayerConnection(d_model, dropout_p=0.1)

    # 3- 演示子层的组合
    # 3.1- 多头自注意力层 + 残差连接和层归一化
    # 3.1.1- 创建 多头自注意力层 数据处理规则
    # 定义方式一：函数的形式
    @MPoint(append_message="子层链接的测试")
    def data_handler_fn(data):
        # 创建 多头自注意力层 类的实例对象
        attention_obj = MultiHeadAttention(d_model=d_model, head=8, dropout_p=0.1)

        # 调用forward
        return attention_obj(
            query=data,
            key=data,
            value=data,
        )

    # 3.1.2- 组装得到子层：也就是调用SubLayerConnection它的forward
    multi_head_attention_output = SubLayerConnection_obj(data, data_handler_fn)
    print(f"多头自注意力层输出结果形状：{multi_head_attention_output.shape}")

    # 3.2- 前馈网络层 + 残差连接和层归一化
    # 定义方式二：lambda的形式
    feed_forward_output = SubLayerConnection_obj(
        multi_head_attention_output,
        lambda data: FeedForward(d_model=d_model, output_dim=1024, dropout_p=0.1)(data),
    )
    print(f"前馈网络层输出结果形状：{feed_forward_output.shape}")


#################################################################################################


# 前馈网络/前馈全连接层：通过调整张量大小，实现强化信息的过程
class FeedForward(nn.Module):
    def __init__(self, d_model, output_dim, dropout_p=0.1):
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建网络结构
        # 2.1- 第一个线性层：用来对输入的词向量维度进行放大，例如：512->1024
        self.linear_1 = nn.Linear(in_features=d_model, out_features=output_dim)

        # 2.2- 创建ReLU激活函数
        # 修改原因：原代码只有Linear -> Dropout -> Linear；推理时Dropout关闭，
        # 两个线性层可以合并成一个线性层，前馈网络无法学习复杂的非线性特征。
        self.relu = nn.ReLU()

        # 2.3- 随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

        # 2.4- 第二个线性层：用来对上一个线性层输出的结果进行浓缩，例如：1024->512
        self.linear_2 = nn.Linear(in_features=output_dim, out_features=d_model)

    def forward(self, data):
        # 1- 调用第一个线性层
        data = self.linear_1(data)

        # 2- 通过ReLU引入非线性。
        # 修改原因：经典Transformer前馈网络必须在线性层之间加入非线性激活；
        # Dropout负责正则化，不能替代ReLU的非线性表达能力。
        data = self.relu(data)

        # 3- 调用随机失活，缓解过拟合
        data = self.dropout(data)

        # 4- 调用第二个线性层
        data = self.linear_2(data)

        return data


##################################################################################################

if __name__ == "__main__":
    # use_attention()

    # use_multi_head_attention()

    use_SubLayerConnection()
