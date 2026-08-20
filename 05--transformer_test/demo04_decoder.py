"""
解码器层由如下3层组成
    1- 第一层：掩码多头自注意力层
        掩码多头自注意力 + 层归一化 + 残差连接
    2- 第二层：多头注意力层
        多头注意力 + 层归一化 + 残差连接
    3- 第三层：前馈网络层
        前馈网络 + 层归一化 + 残差连接
"""

from demo03_encoder import *
from demo02_component import *


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        mask_multi_head_self_attn,
        multi_head_attn,
        feed_forward,
        dropout_p=0.1,
    ):
        """
        定义解码器层的结构
        :param d_model: 词向量维度
        :param mask_multi_head_self_attn: 掩码多头自注意力类的实例对象
        :param multi_head_attn: 多头注意力类的实例对象
        :param feed_forward: 前馈网络类的实例对象
        :param dropout_p:
        """
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.mask_multi_head_self_attn = mask_multi_head_self_attn
        self.multi_head_attn = multi_head_attn
        self.feed_forward = feed_forward

        # 3- 创建子层
        # 3.1- 第一个子层：掩码多头自注意力子层
        self.mask_multi_self_layer = SubLayerConnection(self.d_model, self.dropout_p)

        # 3.2- 第二个子层：多头注意力子层
        self.multi_layer = SubLayerConnection(self.d_model, self.dropout_p)

        # 3.3- 第三个子层：前馈网络子层
        self.feed_forward_layer = SubLayerConnection(self.d_model, self.dropout_p)

    def forward(self, data, mask, encoder_output, encoder_mask=None):
        # 1- 数据经过第一个层的处理：掩码多头自注意力子层
        data = self.mask_multi_self_layer(
            data,
            lambda x: self.mask_multi_head_self_attn(
                query=x, key=x, value=x, mask=mask
            ),
        )

        # 2- 数据经过第二个层的处理：多头注意力子层
        data = self.multi_layer(
            data,
            lambda x: self.multi_head_attn(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask=encoder_mask,
            ),
        )

        # 3- 数据经过第三个层的处理：前馈网络子层
        data = self.feed_forward_layer(data, lambda x: self.feed_forward(x))

        return data


# 解码器
class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, N):
        """
        定义解码器结构。也就是将解码器层重复N层
        :param decoder_layer: 解码器层
        :param N:
        """
        # 1- 初始化父类
        super().__init__()

        # 2- 将解码器层创建N对象
        self.decoder_layer_list = clones(decoder_layer, N)

        # 3- 层归一化
        """
            当前子层连接采用Pre-LN结构，因此堆叠N层后保留最终LayerNorm，
            对整个解码器输出进行统一归一化。
        """
        self.layer_norm = LayerNorm(decoder_layer.d_model)

    def forward(self, data, mask, encoder_output, encoder_mask=None):
        """
        注意：for循环中两个data需要与传递给forward的参数名称完全一致，否则6层网络处理的数据就会断开
        """

        for decoder_layer in self.decoder_layer_list:
            data = decoder_layer(data, mask, encoder_output, encoder_mask)

        return self.layer_norm(data)


# 测试解码器
# 测试最终的解码器
@MPoint(append_message="测试解码器")
def use_decoder():
    # 常用参数定义
    d_model = 512
    dropout = 0.1

    # 1- 获得编码器最终结果数据。作为key、value传递给到解码器
    encoder_output = use_encoder()

    # 2- 准备解码器端原始目标值数据
    y = torch.tensor(
        [
            # 例如法语单词索引
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    # 3- 构建解码器
    # 3.1- 基础组件实例对象创建
    # 解码器端的词嵌入层
    embed = Embedding(vocab_size=1200, d_model=d_model)

    # 解码器端的位置编码
    position_encoding = PositionalEncoding(d_model=d_model, dropout_p=dropout)

    # 前馈网络子层
    feed_forward = FeedForward(d_model=d_model, output_dim=1024, dropout_p=dropout)

    # 多头注意力子层
    src_attn = MultiHeadAttention(d_model=d_model, head=8, dropout_p=dropout)

    # 掩码多头自注意力子层
    self_attn = MultiHeadAttention(d_model=d_model, head=8, dropout_p=dropout)

    # 3.2- 拼接得到解码器层
    decoder_layer = DecoderLayer(
        d_model=d_model,
        mask_multi_head_self_attn=self_attn,
        multi_head_attn=src_attn,
        feed_forward=feed_forward,
        dropout_p=dropout,
    )

    # 3.3- 解码器层堆叠N层，得到解码器
    decoder = Decoder(decoder_layer, 6)

    # 4- 处理数据
    # 4.1- 准备掩码
    # 当前注意力实现约定1/True表示允许关注，因此使用下三角因果掩码。
    # 二维掩码会在MultiHeadAttention中自动广播到所有批次和所有头。
    seq_len = y.shape[1]
    mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.bool))

    # 4.2- 得到词向量+位置编码
    data = embed(y)
    data = position_encoding(data)

    # 4.3- 调用解码器
    decoder_result = decoder(data, mask, encoder_output)

    print(f"解码器最终的结果：{decoder_result.shape}")


if __name__ == "__main__":
    use_decoder()
