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

        # 2- 教师模型的：池化层，实际就是nn.Linear+激活函数。不用额外定义
        """
        1- last_hidden_state[:,0]和pooler_output的区别。
        区别：需要对last_hidden_state[:,0]经过nn.Linear和激活函数处理后，才能得到pooler_output
        对应源代码位置：BertModel文件的697行

        2- 为什么使用pooler_output，而不使用last_hidden_state[:,0]
        使用pooler_output的原因有如下几个
            语义对齐：pooler_output已经是句子级别的表示，与下游任务的张量形状是对其的
        如果不做蒸馏，那么用last_hidden_state[:,0]；如果做模型蒸馏用pooler_output。推荐全部都用pooler_output

        3- 获得池化层后的结果有两种方式：
            3.1- 方式一：推荐。通过实例属性获得 bert_output.pooler_output
            3.2- 方式二：通过实例属性索引获得 bert_output[1]。1的原因是pooler_output是类中的第2个实例属性
                        对应源代码位置：BertModel文件的1017行
        """

        # 下面两个代码的作用完全相同
        pooler_output = bert_output.pooler_output
        # pooler_output = bert_output[1]

        # 3- 教师模型的：线性层
        return self.linear(pooler_output)
