import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn

# HuggingFace 提供的据集加载工具, 可以加载本地数据, 也可以加载公开数据源
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入BERT相关组件(中文文本分词器, 预训练的BERT模型)
from transformers import BertTokenizer, BertModel

# 终端打印美化
from rich import print

############################################################################################

# 指定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1- 创建预训练模型的实例对象
model_path = r"PretrainedModel/bert-base-chinese"
bert_tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path).to(device)

predict_mask_index = 16  # 要对句子中索引为多少的词进行填空

############################################################################################

# 2- 每个批次数据的具体处理
"""
    data参数的数据结构：
        [
            {"label":值,"text":值},
            {"label":值,"text":值}
            ...
        ]
"""


def collate_fn(data):
    # 1- 获取每条样本的句子内容
    sents = [line["text"] for line in data]

    # 2- 批量编码处理，得到词索引组成的张量
    data_tensor = bert_tokenizer(
        sents, return_tensors="pt", max_length=32, padding=True, truncation=True
    )

    # 3- 提取各个字段
    input_ids = data_tensor["input_ids"]
    token_type_ids = data_tensor["token_type_ids"]
    attention_mask = data_tensor["attention_mask"]

    # 4- 将索引位置为16替换成[MASK]
    # 4.1- 先将原来的索引位置为16的词索引保留下来，作为目标值
    # labels的格式 [第一条句子的第16个词,第二条句子的第16个词..]
    labels = input_ids[:, predict_mask_index].clone()
    # print("input_ids-->", input_ids)
    # print("input_ids-->", input_ids[:, 16])

    # 4.2- 将[MASK]对应的词索引，重新填充回每条句子的索引位置为16的地方
    # 4.2.1- 获取[MASK]对应的词索引
    # print("修改前-->", input_ids)
    mask_index = bert_tokenizer.get_vocab()[bert_tokenizer.mask_token]
    input_ids[:, predict_mask_index] = mask_index
    # print("修改后-->", input_ids)
    # print(mask_index)  # 103
    # print(bert_tokenizer.mask_token)  # [MASK]

    # 5- 返回结果
    return input_ids, token_type_ids, attention_mask, labels


# 3- 获得数据加载器
def get_dataloader(task_type):
    # 1- 读取文件
    data_files = {"train": "train.csv", "test": "test.csv"}
    data_set = load_dataset(path="data", data_files=data_files, split=task_type)

    # 2- 数据过滤
    """
        1- 因为没有现成的[MASK]文件，因此人为选择索引为16的位置对应的词替换成[MASK]
        2- 所以句子的完整长度至少需要超过17，我们本次选择范围要求是句子的长度>32
        3- 不是非得选32，可以设置为其他的。
    """
    data_set = data_set.filter(lambda line: len(line["text"]) > 32)

    # 3- 创建Dataloader实例对象
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader


# 4- 自定义微调网络结构
class AiModel(nn.Module):
    def __init__(self):
        # 1- 初始化父类
        super().__init__()

        # 2- 创建线性层
        # 方式一：随机初始化偏置
        # self.linear = nn.Linear(768,bert_tokenizer.vocab_size)

        # 方式二：推荐，手动对偏置进行全0初始化，可以让模型训练更加稳定
        self.linear = nn.Linear(768, bert_tokenizer.vocab_size, bias=False)
        self.linear.bias = nn.Parameter(torch.zeros(bert_tokenizer.vocab_size))

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 1- 先调用预训练模型：不允许预训练模型进行梯度下降计算，防止更新参数
        with torch.no_grad():
            output = bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        # 2- 再调用我们自己定义的微调网络层：因为是针对某个具体位置的词进行预测，因此取要预测词对应位置的隐藏状态信息
        output = self.linear(output.last_hidden_state[:, predict_mask_index])

        return output


# 5- 模型训练
def train_model():
    # 1- 创建自定义微调网络结构
    model = AiModel().to(device)

    # 2- 禁用预训练模型的梯度下降，防止更新参数
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 3- 创建相关实例对象
    # 3.1- 损失函数
    loss = nn.CrossEntropyLoss()

    # 3.2- 优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # 4- 循环训练
    # 4.1- 设置模式为训练模型：也就是允许Dropout随机失活
    model.train()
    epochs = 3

    # 4.2- 训练
    for epoch in range(epochs):
        dataloader = get_dataloader(task_type="train")
        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(
            tqdm(dataloader), start=1
        ):
            # 4.3- 将数据发送到指定设备
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 4.4- 前向传播
            output = model(input_ids, token_type_ids, attention_mask)
            # 4.5- 算损失值
            loss_value = loss(output, labels)
            # 4.6- 反向传播
            optimizer.zero_grad()
            loss_value.sum().backward()
            optimizer.step()

            # 5- 阶段性打印相关信息
            if i % 20 == 0:
                # 5.1- 获取预测概率最高的那个索引
                tmp = torch.argmax(output, dim=-1)
                # 5.2- 统计准确率
                acc = (tmp == labels).sum().item() / len(labels)
                print(
                    f"轮次{epoch+1}，第{i}批次，当前的损失值是{loss_value}，准确率{acc}"
                )

        # 6- 每个epoch保存一次训练好的模型
        torch.save(model.state_dict(), f"model/fill_mask_{epoch+1}.pkl")

