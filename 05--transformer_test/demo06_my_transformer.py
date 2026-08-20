"""
Transformer框架的结构：
    编码器端_输入部分
        词嵌入层
        位置编码

    编码器
        多头自注意力子层：多头自注意力 + 层归一化 + 残差连接
        前馈网络子层：前馈网络 + 层归一化 + 残差连接

    解码器端_输入部分
        词嵌入层
        位置编码

    解码器
        掩码多头自注意力子层：掩码多头自注意力 + 层归一化 + 残差连接
        多头注意力子层（交叉注意力）：多头注意力 + 层归一化 + 残差连接
        前馈网络子层：前馈网络 + 层归一化 + 残差连接

    输出部分
        线性层
        Softmax激活函数

    结果
"""

from demo04_decoder import *
from demo05_output import *


# 创建Transformer框架类
class MyTransformer(nn.Module):
    def __init__(self, en_embed_pos, encoder, de_embed_pos, decoder, output):
        """
        将前面开发的各个部分组织得到完整的Transformer框架
        :param en_embed_pos: 编码器端_输入部分 实例对象
        :param encoder: 编码器 实例对象
        :param de_embed_pos: 解码器端_输入部分 实例对象
        :param decoder: 解码器 实例对象
        :param output: 输出部分 实例对象
        """
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.en_embed_pos = en_embed_pos
        self.encoder = encoder
        self.de_embed_pos = de_embed_pos
        self.decoder = decoder
        self.output = output

    def forward(self, en_input, de_input, mask, src_mask=None):
        """
        Transformer前向传播
        :param en_input: 输入到编码器端的词索引数据
        :param de_input: 输入到解码器端的词索引数据
        :param mask: 解码器端的因果掩码/目标端掩码
        :param src_mask: 编码器端PAD掩码；没有PAD时可以传None
        :return: 未经Softmax处理的logits，形状[batch_size,target_len,target_vocab_size]
        """
        # 1- 编码器端_输入部分：词嵌入层、位置编码
        encoder_data = self.en_embed_pos(en_input)

        # 2- 编码器
        # 修改原因：src_mask用于屏蔽源句中的PAD；如果不向编码器传递，
        # 编码器自注意力会把PAD位置也当作正常单词参与计算。
        encoder_data = self.encoder(encoder_data, src_mask)

        # 3- 解码器端_输入部分：词嵌入层、位置编码
        decoder_data = self.de_embed_pos(de_input)

        # 4- 解码器
        # 修改原因：解码器既需要mask防止偷看目标端未来词，也需要src_mask在交叉注意力中
        # 屏蔽源端PAD，所以两份mask都要继续传给Decoder。
        decoder_data = self.decoder(
            data=decoder_data,
            mask=mask,
            encoder_output=encoder_data,
            encoder_mask=src_mask,
        )

        # 5- 输出部分
        return self.output(data=decoder_data)


# 调用Transformer框架
@MPoint(append_message="调用Transformer框架")
def get_my_transformer():
    # 常用变量
    d_model = 512
    dropout_p = 0.1
    de_vocab_size = 4345

    # -------------------------------------------- 编码器部分 ---------------------------------------------
    # 词嵌入层
    en_ebd = Embedding(vocab_size=1000, d_model=d_model)

    # 位置编码
    en_pos = PositionalEncoding(d_model=d_model, dropout_p=dropout_p, max_len=60)

    # 多头自注意力
    en_multi_self_attn = MultiHeadAttention(
        d_model=d_model, head=8, dropout_p=dropout_p
    )

    # 前馈网络
    en_ff = FeedForward(d_model=d_model, output_dim=1024, dropout_p=dropout_p)

    # 组装得到编码器层
    encoder_layer = EncoderLayer(
        d_model=d_model,
        multi_head_self_attn=en_multi_self_attn,
        feed_forward=en_ff,
        dropout_p=dropout_p,
    )

    # 得到编码器
    encoder = Encoder(encoder_layer=encoder_layer, N=6)

    # ----------------------------------------------- 解码器部分 ---------------------------------------
    # 词嵌入层
    de_ebd = Embedding(vocab_size=de_vocab_size, d_model=d_model)

    # 位置编码
    de_pos = PositionalEncoding(d_model=d_model, dropout_p=dropout_p, max_len=60)

    # 掩码多头自注意力
    de_multi_self_attn = MultiHeadAttention(
        d_model=d_model, head=8, dropout_p=dropout_p
    )

    # 多头注意力
    de_multi_attn = MultiHeadAttention(d_model=d_model, head=8, dropout_p=dropout_p)

    # 前馈网络
    de_ff = FeedForward(d_model=d_model, output_dim=1024, dropout_p=dropout_p)

    # 组装得到编码器层
    decoder_layer = DecoderLayer(
        d_model=d_model,
        mask_multi_head_self_attn=de_multi_self_attn,
        multi_head_attn=de_multi_attn,
        feed_forward=de_ff,
        dropout_p=dropout_p,
    )

    # 得到编码器
    decoder = Decoder(decoder_layer=decoder_layer, N=6)

    # ---------------- 输出部分 ----------------
    output = Output(d_model=d_model, vocab_size=de_vocab_size)

    # ---------------- 组装得到Transformer框架 ----------------
    my_transformer = MyTransformer(
        en_embed_pos=nn.Sequential(en_ebd, en_pos),
        encoder=encoder,
        de_embed_pos=nn.Sequential(de_ebd, de_pos),
        decoder=decoder,
        output=output,
    )

    # 修改原因：clones使用deepcopy后，各层不仅结构相同，初始参数值也完全相同；
    # 同时Embedding和多个Linear需要合适的初始数值范围，才能让训练更加稳定。
    # 因此对二维及以上参数统一执行Xavier初始化。
    # 这样不仅能控制Embedding和线性层的初始数值范围，也能让deepcopy得到的各层
    # 从不同随机参数开始训练；LayerNorm的一维k、b仍保持1和0。
    for parameter in my_transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    print(f"框架结构信息：{my_transformer}")
    return my_transformer


# 测试Transformer框架
@MPoint(append_message="测试Transformer框架")
def use_my_transformer():
    # 1- 获得我们自己的Transformer框架的实例对象
    my_transformer = get_my_transformer()

    # 2- 准备数据：以前面的英译法案例为例讲解
    # 2.1- 输入到编码器端的原始数据：也就是英语句子
    en_input = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    # 2.2- 输入到解码器端的数据必须是目标句子右移后的结果。
    # 修改原因：训练时如果把未右移的完整目标句直接输入，当前位置就已经包含正确答案；
    # 因果mask只能屏蔽未来位置，无法屏蔽当前位置本身，因此必须先右移。
    # 这里假设0是BOS：训练时输入[BOS,词1,词2,词3]，标签则是[词1,词2,词3,EOS]。
    de_input = torch.tensor([[0, 223, 2344, 456], [0, 2456, 131, 456]])

    # 2.3- 解码器的掩码
    # 修改原因：本项目约定True表示允许关注，因此必须使用下三角矩阵；
    # 这样第t个位置只能看到0~t位置，不能看到t后面的未来词。
    seq_len = de_input.shape[1]
    mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.bool))

    # 3- 调用Transformer框架
    # 修改原因：训练时把logits直接交给CrossEntropyLoss；只有在展示或预测时才转成概率，
    # 避免在模型和损失函数中重复执行Softmax。
    logits = my_transformer(en_input, de_input, mask)
    probabilities = my_transformer.output.probabilities(logits)

    print(f"最终logits的张量形状：{logits.shape}")
    print(f"第1条句子第1个位置的概率和：{probabilities[0, 0].sum()}")


if __name__ == "__main__":
    use_my_transformer()
