import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self, query_size, key_size, value_size, weighted_size, c_size, input_size
    ) -> None:
        """
        query_size：张量Q最后一维的形状，目前是8
        key_size：张量K最后一维的形状，目前是8
        value_size：张量V最后一维的形状，目前是8
        weighted_size：相似性和权重张量中最后一维的形状，目前是5
        c_size：中间语义张量C中最后一维的形状，目前是8
        input_size：输入到解码器Decoder的GRU中最后一维的形状，目前是8
        """
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.weighted_size = weighted_size
        self.c_size = c_size
        self.input_size = input_size

        # 3- 搭建网络模型结构
        # 3.1- 第一个线性层：对应原理图中的第2步，用来计算Q和K的相似性
        """
            参数解释：
                in_features：输入的特征个数
                out_features：输出的特征个数
        """
        self.atten_linear = nn.Linear(
            in_features=self.query_size + self.key_size, out_features=self.weighted_size
        )
        # self.atten_linear = nn.Linear(in_features=16,out_features=5)

        # 3.2- 第二个线性层：对应原理图中的第6步，将张量形状调整为解码器端要求输入的形状
        self.combine_linear = nn.Linear(
            in_features=self.query_size + self.c_size, out_features=self.input_size
        )
        # self.combine_linear = nn.Linear(in_features=16,out_features=8)

    def forward(self, Q, K, V):
        # 1- Q和K拼接
        # dim=-1：表示对张量的最后一维进行拼接，其他维保持不变
        qk_concat = torch.concat(tensors=(Q, K), dim=-1)
        print("qk_concat", qk_concat.shape)

        # 2- Q和K算相似性
        score = self.atten_linear(qk_concat)
        print("score", score.shape)

        # 3- 相似性转成权重
        weighted = torch.softmax(score, dim=-1)
        print("weighted", weighted.shape)

        # 4- 权重矩阵和V进行矩阵乘法运算，得到专属信息包C
        C = torch.bmm(weighted, V)
        print("C", C.shape)

        # 5- Q和C拼接
        qc_concat = torch.concat(tensors=(C, Q), dim=-1)
        print("qc_concat", qc_concat.shape)

        # 6- 对Q和C拼接后的结果进行形状调整，达到满足GRU的输入要求
        decoder_input = self.combine_linear(qc_concat)
        print("decoder_input", decoder_input.shape)

        return decoder_input, weighted


if __name__ == "__main__":
    # 因为只讲PyTorch中注意力是如何计算得到。因此拼接后的value、Q、V我们直接手动初始化

    # 1- 定义基础变量
    batch_size = 1  # 每个批次中句子的条数
    seq_len = 5  # 每个句子中词的个数
    hidden_size = 8  # 我们人为设置词向量维度和隐藏向量维度相同
    num_layers = 1  # 隐藏层层数

    # 2- 初始化Q、K、V
    Q = torch.randn(size=(num_layers, batch_size, hidden_size))
    K = torch.randn(size=(num_layers, batch_size, hidden_size))
    V = torch.randn(size=(batch_size, seq_len, hidden_size))  # 多个词的隐藏状态拼接后的
    print("-" * 50)
    print(f"Q--> {Q.shape}\n {Q}")
    print("-" * 50)
    print(f"K--> {K.shape}\n {K}")
    print("-" * 50)
    print(f"V--> {V.shape}\n {V}")

    # 3- 计算注意力
    query_size = hidden_size
    key_size = hidden_size
    value_size = hidden_size
    weighted_size = seq_len
    c_size = hidden_size
    input_size = hidden_size

    attn_model = Attention(
        query_size, key_size, value_size, weighted_size, c_size, input_size
    )

    decoder_input, attn_weight = attn_model(Q, K, V)

    print(f"权重值{attn_weight}")
    print(f"权重值和{attn_weight.sum()}")
    print(f"decoder_input-->{decoder_input.shape}")  # [1,1,8]
