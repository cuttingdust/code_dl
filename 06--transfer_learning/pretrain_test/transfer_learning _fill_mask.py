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
