# 教师模型：Bert预训练模型
import torch
import torch.nn as nn

from transformers import BertModel
from transformers import BertConfig
from transformers import BertTokenizer

from config import Config

config = Config()


class BertTeacherModel(nn.Module):
    def __init__(self):
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建网络结构
        # 2.1- 先定义Bert模型
        self.bert_model = BertModel.from_pretrained(config.bert_path)

        # 禁用预训练模型的反向传播
        for param in self.bert_model.parameters():
            param.requires_grad_(False)

        # 2.2- 再定义我们自己的网络结构
        in_features = BertConfig.from_pretrained(config.bert_path).hidden_size  # 768
        self.linear = nn.Linear(
            in_features=in_features, out_features=config.classname_len
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # torch.no_grad()冻结bert的反向传播。如果放开，训练耗时大量增加
        # 1- 教师模型的：嵌入层（词嵌入层、片段编码、位置编码）、Encoder编码器
        with torch.no_grad():
            bert_output = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # 2- 教师模型的：池化层，实际就是nn.Linear + Tanh激活函数，不用额外定义
        """
        1- last_hidden_state[:, 0, :]和pooler_output的区别

            last_hidden_state[:, 0, :]是[CLS]位置未经池化层处理的特征。

            pooler_output的计算过程是：
                last_hidden_state[:, 0, :]
                    -> BertPooler中的Linear层
                    -> Tanh激活函数
                    -> pooler_output

            对应源代码可以查看BertModel和BertPooler的forward方法。
            注意：Transformers版本不同，具体源码行号可能发生变化。

        2- 为什么这里使用pooler_output

            当前任务是对整条新闻文本进行分类，需要获得句子级别的特征。
            pooler_output的形状是[batch_size, hidden_size]，它已经把[CLS]位置
            进一步转换成固定长度的句子级表示，因此可以交给后面的Linear层分类。

            注意：是否使用pooler_output，与是否进行模型蒸馏没有必然关系。
            不做蒸馏时可以使用pooler_output，做蒸馏时也可以使用
            last_hidden_state[:, 0, :]。

            真正重要的是：分类层训练、模型蒸馏和模型推理必须使用同一种特征。
            如果Linear分类层是使用last_hidden_state[:, 0, :]训练的，就必须继续
            使用last_hidden_state[:, 0, :]；不能直接改接pooler_output。

            当前蒸馏实验会重新训练教师模型的Linear分类层，所以这里可以使用
            pooler_output，让新的Linear分类层重新学习对应的分类边界。

        3- 获得池化层结果的两种方式

            3.1- 方式一（推荐）：通过实例属性获得，可读性更好
                pooler_output = bert_output.pooler_output

            3.2- 方式二：通过索引获得。pooler_output是返回结果中的第2项
                pooler_output = bert_output[1]
        """

        # 下面两行代码作用相同，推荐使用属性名称，代码含义更加清楚。
        pooler_output = bert_output.pooler_output
        # pooler_output = bert_output[1]

        # 3- 教师模型的：线性层
        return self.linear(pooler_output)
