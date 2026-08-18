import torch
from torch import nn

if __name__ == "__main__":
    # 1- 创建LSTM循环网络层
    """
    参数解释：
        input_size：输入数据的向量维度，也就是向量中有多少个数字
        hidden_size：隐藏层隐藏状态的向量维度
        num_layers：隐藏层的层数
        bidirectional：是否是双向的网络层。False表示单向；True表示双向，也就是Bi-LSTM
    """
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=5,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
    )

    # 2- 准备数据
    # 2.1- 本次时间步输入的数据
    input = torch.randn(size=(6, 3, 4))

    # 2.2- 上一个时间步的隐藏状态。第一个时间步的隐藏状态一般用全零初始化
    h0 = torch.zeros(size=(1, 3, 5))

    # 2.3- 初始的细胞状态，一般用全零初始化
    c0 = torch.zeros(size=(1, 3, 5))

    # 3- 调用LSTM
    """
        参数解释：
            输入参数：
                input：本次时间步输入的数据，张量形状是：[seq_len每条句子中词的个数,batch_size每个批次中句子的条数,input_size输入数据的向量维度]
                h0：上一个时间步的隐藏状态，张量形状是：[num_layers隐藏层的层数,batch_size每个批次中句子的条数,hidden_size隐藏层隐藏状态的向量维度]
                    第一个时间步的隐藏状态一般用全零初始化
                c0：上一个时间步的细胞状态，张量形状是：[num_layers隐藏层的层数,batch_size每个批次中句子的条数,hidden_size隐藏层隐藏状态的向量维度]
                    初始的细胞状态一般用全零初始化

                注意：h0和c0需要使用元组包起来

            返回结果：
                output：本次时间步的预测结果，张量形状是：[seq_len每条句子中词的个数,batch_size每个批次中句子的条数,hidden_size隐藏层隐藏状态的向量维度]
                hn：本次时间步的隐藏状态，张量形状是：[num_layers隐藏层的层数,batch_size每个批次中句子的条数,hidden_size隐藏层隐藏状态的向量维度]
                c：更新后的细胞状态，张量形状是：[num_layers隐藏层的层数,batch_size每个批次中句子的条数,hidden_size隐藏层隐藏状态的向量维度]

                注意：hn和c需要使用元组包起来
    """
    output, (hn, c) = lstm(input, (h0, c0))

    # 4- 查看结果
    print("-" * 50)
    print(f"output预测结果_形状：{output.shape}")  # [6,3,5]

    print("-" * 50)
    print(f"hn更新后的隐藏状态_形状：{hn.shape}")  # [1,3,5]
    print(f"c更新后的细胞状态_形状：{c.shape}")  # [1,3,5]

    print("-" * 50)
    print(f"output预测结果_内容：{output}")
    print(f"hn更新后的隐藏状态_内容：{hn}")
    print(f"c更新后的细胞状态_内容：{c}")  # 不用关心内部的值
