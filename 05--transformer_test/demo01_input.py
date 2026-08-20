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
import time
import functools
import inspect

####################################################################################################


class MPoint:
    """
    Python版本的函数执行跟踪器，作用类似C++项目中的：

        #define MPoint MTracePoint point(__FUNCTION__)

    C++版本利用局部对象的构造函数和析构函数，在进入、离开函数时自动打印日志。
    Python没有完全相同的宏和确定性析构机制，因此这里使用两种Python原生机制实现：

    1. 装饰器（推荐用于跟踪整个函数）

        @MTracePoint()
        def getdata():
            ...

    2. 上下文管理器（用于只跟踪函数内部的一段代码）

        with MTracePoint("load_data", "读取训练数据"):
            ...

    日志会自动包含：函数名、可选附加消息、执行结果和耗时。
    如果函数内部抛出异常，异常不会被吞掉，只会先记录异常类型后继续向外抛出。
    """

    def __init__(self, function_name=None, append_message=None):
        self.function_name = function_name
        self.append_message = append_message
        self.start_time = None

    def __call__(self, func):
        """让MTracePoint对象可以作为装饰器使用。"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 每次函数调用都创建一个新的跟踪对象，避免递归或多线程调用时共享开始时间。
            with type(self)(
                function_name=self.function_name or func.__qualname__,
                append_message=self.append_message,
            ):
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        """进入with代码块时执行，等价于C++ MTracePoint构造函数中的logStart。"""
        if self.function_name is None:
            # inspect.currentframe()得到当前__enter__栈帧，f_back是with所在函数的栈帧。
            caller_frame = inspect.currentframe().f_back
            self.function_name = caller_frame.f_code.co_name

        extra_message = f" {self.append_message}" if self.append_message else ""
        print(f"=== BEGIN === {self.function_name}{extra_message} Start!")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """离开with代码块时执行，等价于C++ MTracePoint析构函数中的logEnd。"""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        extra_message = f" {self.append_message}" if self.append_message else ""

        if exc_type is None:
            result = "成功"
        else:
            result = f"异常：{exc_type.__name__}: {exc_value}"

        print(
            # BEGIN有5个字符，END只有3个字符，因此在END后补3个空格，
            # 让BEGIN和END后面的===以及函数名从同一列开始。
            f"=== END   === {self.function_name}{extra_message} End! "
            # f"[{result}，耗时 {elapsed_ms:.3f} ms]"
        )

        # 返回False表示不吞掉异常：如果被跟踪代码出错，程序仍然按正常方式抛出异常。
        return False


####################################################################################################


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.vocab_size = vocab_size  # 词汇表大小
        self.d_model = d_model  # 词向量的维度。例如：512

        # 3- 搭建网络结构：只有一个词嵌入层
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # 4- 初始化词嵌入权重
        # 修改原因：nn.Embedding默认初始化的标准差接近1，后面再乘sqrt(d_model)会让数值过大，
        # 从而淹没数值范围只有[-1, 1]的位置编码。
        # 这里先把标准差缩小到1/sqrt(d_model)，经过forward中的缩放后约为1。
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model**-0.5)

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
    def __init__(self, d_model, dropout_p=0.1, max_len=60):
        """
        位置编码初始化方法
        :param d_model: 词向量维度。例如：512
        :param dropout_p: 神经元随机失活概率
        :param max_len: 能够处理的句子最大长度。也就是词的个数
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 创建随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

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
        # 修改原因：当d_model是奇数时，奇数列的数量会比偶数列少1；
        # 如果不按实际列数截取，给cos位置编码赋值时会发生张量形状不一致。
        pe[:, 1::2] = torch.cos(position_value[:, : pe[:, 1::2].shape[1]])

        # 8- 调整pe的形状变成3维，也就是[60,d_model]->[1,60,d_model]每个批次1条句子，每个句子最多60个词，词向量是d_model
        pe = pe.unsqueeze(0)

        # 9- 将固定位置编码注册为buffer。
        # 修改原因：位置编码是固定公式生成的数据，不应该被优化器当作权重训练；
        # 但它仍然需要跟随模型移动设备并保存，所以不能只是普通成员变量。它会：
        #    1. 跟随模型在CPU和GPU之间移动；
        #    2. 保存到state_dict中；
        #    3. 可以通过self.pe访问。
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
            如果句子长度超过max_len，位置编码和词向量无法按位置相加，因此应明确报错，
            而不是悄悄丢弃后面的词。
        """
        # 修改原因：原代码在句子长度超过max_len时，会在张量相加处抛出难懂的尺寸错误；
        # 这里提前检查并告诉使用者应该增大max_len。
        if embed.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"输入句子长度{embed.shape[1]}超过位置编码支持的最大长度"
                f"{self.pe.shape[1]}，请增大max_len"
            )

        result = embed + self.pe[:, : embed.shape[1]]
        # print(f"对应的位置编码值：{self.pe[:, :embed.shape[1]]}")
        return self.dropout(result)


@MPoint(append_message="测试位置编码")
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
    my_pe = PositionalEncoding(d_model=d_model, dropout_p=0.1, max_len=60)

    # 调用位置编码，最终效果是往词向量中加上了位置编码的值
    result = my_pe(word_embed)
    print(f"最终的形状：{result.shape}")
    print(f"最终的值：{result}")

    return result


# 可视化位置编码
def plot_position():
    # 1. 实例化位置编码器.
    # 修改原因：可视化的目标是观察固定的位置编码曲线；如果保持训练模式的Dropout，
    # 部分位置编码会被随机置0，每次画出的曲线也可能不同，因此这里关闭Dropout。
    my_position = PositionalEncoding(d_model=20, dropout_p=0.0, max_len=100)
    my_position.eval()

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
