import torch
import torch.nn as nn


def single_layer():
    # 1- 创建GRU循环网络层
    """
    参数解释：
        input_size：输入数据的词向量维度
        hidden_size：隐藏层中隐藏状态的向量维度
        num_layers：隐藏层的层数。默认是1
        bidirectional：默认是False，也就是单向的GRU。如果为True，就是双向的GRU
    """
    gru = nn.GRU(input_size=4, hidden_size=5, num_layers=1, batch_first=False)

    # 2- 准备数据
    # 2.1- 本次的输入数据
    """
        张量形状要求：
            [
                seq_len每条句子中有多少个词,
                batch_size每个批次中句子的条数有多少条,
                input_size输入数据的词向量维度
            ]
    """
    input = torch.randn(size=(6, 3, 4))

    # 2.2- 上一个时间步的隐藏状态。最初始的时候，一般进行全0初始化
    """
        张量形状要求：
            [
                num_layers隐藏层的层数,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]
    """
    h0 = torch.zeros(size=(1, 3, 5))

    # 3- 调用GRU
    """
        输出的output张量的形状：
            [
                seq_len每条句子中有多少个词,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]

        输出的hidden张量的形状：
            [
                num_layers隐藏层的层数,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]
    """
    output, hidden = gru(input, h0)

    # 4- 打印结果
    print("-" * 50)
    print(f"output形状-->{output.shape}")  # 6,3,5
    print(f"hidden形状-->{hidden.shape}")  # 2,3,5

    print("-" * 50)
    print(f"output-->{output}")
    print(f"hidden-->{hidden}")


def double_layer():
    # 1- 创建GRU循环网络层
    """
    参数解释：
        input_size：输入数据的词向量维度
        hidden_size：隐藏层中隐藏状态的向量维度
        num_layers：隐藏层的层数。默认是1
        bidirectional：默认是False，也就是单向的GRU。如果为True，就是双向的GRU
    """
    gru = nn.GRU(input_size=4, hidden_size=5, num_layers=1, bidirectional=True)

    # 2- 准备数据
    # 2.1- 本次的输入数据
    """
        张量形状要求：
            [
                seq_len每条句子中有多少个词,
                batch_size每个批次中句子的条数有多少条,
                input_size输入数据的词向量维度
            ]
    """
    input = torch.randn(size=(6, 3, 4))
    # 2.2- 上一个时间步的隐藏状态。最初始的时候，一般进行全0初始化
    """
        张量形状要求：
            [
                num_layers隐藏层的层数,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]
    """
    h0 = torch.zeros(size=(2, 3, 5))

    # 3- 调用GRU
    """
        输出的output张量的形状：
            [
                seq_len每条句子中有多少个词,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]

        输出的hidden张量的形状：
            [
                num_layers隐藏层的层数,
                batch_size每个批次中句子的条数有多少条,
                hidden_size隐藏层中隐藏状态的向量维度
            ]
    """
    output, hidden = gru(input, h0)

    # 4- 打印结果
    print("-" * 50)
    print(f"output-->{output.shape}")
    print(f"hidden-->{hidden.shape}")

    print("-" * 50)
    print(f"output-->{output}")
    print(f"hidden-->{hidden}")


if __name__ == "__main__":
    # 单向GRU
    # single_layer()

    # 双向GRU
    double_layer()
