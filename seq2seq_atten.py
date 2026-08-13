"""
英译法示例
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

# 1- 定义变量
# 运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 翻译开始标识的索引
SOS_TOKEN = 0
# 翻译结束标识的索引
EOS_TOKEN = 1
# 文件路径
file_path = os.path.join(r"data", r"eng-fra-v2.txt")
# 句子长度规范中句子的最大长度
MAX_LENGTH = 10


####################################################################################################


# 2- 数据清洗
def normalize_string(line):
    # 全部转小写，并且去除前后的空白字符
    line = line.lower().strip()

    # 在标点符号的前面增加空格
    line = re.sub(r"([.!?])", r" \1", line)

    # 去除特殊内容（除了26个字母和.!?），替换成空格
    line = re.sub(r"[^a-z.!?]+", r" ", line)

    return line


# 3- 数据预处理
def getdata():
    print("-" * 50)
    # 1- 读取文件的所有行
    """
    大文件推荐用readline
    小文件用readline、readlines都行
    """
    with open(file_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    # print(lines)

    # 2- 循环遍历每一行，拆分得到英语句子和法语句子。得到嵌套列表，格式如下
    # [["英语句子1","法语句子1"], ["英语句子2","法语句子2"]]
    # 普通版
    sen_pairs = []
    for line in lines:
        eng_fre = line.split("\t")

        # tmp_pair的格式是：["英语句子","法语句子"]
        tmp_pair = []
        for sen in eng_fre:
            tmp_pair.append(normalize_string(sen))

        sen_pairs.append(tmp_pair)

    # 简洁版：理解
    # sen_pairs = [[normalize_string(sen) for sen in line.split("\t")] for line in lines]
    print(sen_pairs[:5])

    # 3- 分词
    # 3.1- 初始设置
    english_word2index = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN}
    english_word_n = 2
    french_word2index = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN}
    french_word_n = 2

    # 3.2- 分别对英语句子、法语句子进行分词
    for eng_fre in sen_pairs:
        # 英语
        for word in eng_fre[0].split(" "):
            # 去重处理
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1

        # 法语
        for word in eng_fre[1].split(" "):
            # 去重处理
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # print(english_word2index)

    # 3.3- 将3.2中的词典（key是单词，value是索引）改成key是索引，value是单词的形式
    english_index2word = {value: key for key, value in english_word2index.items()}
    french_index2word = {value: key for key, value in french_word2index.items()}

    return (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        sen_pairs,
    )


(
    english_word2index,
    english_index2word,
    english_word_n,
    french_word2index,
    french_index2word,
    french_word_n,
    sen_pairs,
) = getdata()


# 4- 自定义数据集Dataset
class MyPairsDataset(Dataset):
    def __init__(self, sen_pairs):
        super().__init__()
        # 设置属性值
        self.sen_pairs = sen_pairs
        self.sample_cnt = len(
            self.sen_pairs
        )  # 获得样本条数，也就是英语和法语句子对有多少对

    def __len__(self):
        # 获得样本条数
        return self.sample_cnt

    def __getitem__(self, index):
        """
        根据索引获得对应的样本数据。index是索引，从0开始
        """

        # 1- 防止index出现负数；防止index越界
        index = min(max(index, 0), self.sample_cnt - 1)

        # 2- 获得对应索引的英语句子和法语句子
        x = self.sen_pairs[index][0]  # 英语句子
        y = self.sen_pairs[index][1]  # 法语句子

        # 3- 句子分词转成词索引，最终变成张量
        """
            为什么这里只是增加了句子末尾标识EOS_TOKEN，没有增加句子开始标识SOS_TOKEN？
            答：在seq2seq+注意力机制中，不管是Encoder编码器还是Decoder解码器，都必须明确要有句子末尾标识。
               而开始标识不是必须的，同时我们在后续模型训练的时候再加上开始标识SOS_TOKEN
        """
        # 3.1- 英语句子
        x = [english_word2index[word] for word in x.split(" ")]  # 句子分词转成词索引
        x.append(EOS_TOKEN)  # 列表最后增加句子末尾标识
        x = torch.tensor(x, dtype=torch.long, device=device)  # 变成张量

        # 3.2- 法语句子
        y = [french_word2index[word] for word in y.split(" ")]  # 句子分词转成词索引
        y.append(EOS_TOKEN)  # 列表最后增加句子末尾标识
        y = torch.tensor(y, dtype=torch.long, device=device)  # 变成张量

        return x, y


# 5- 创建Dataloader
def get_dataloader():
    # 1- 创建Dataset
    dataset = MyPairsDataset(sen_pairs)

    # 2- 创建Dataloader
    # 因为在自定义MyPairsDataset我们并没有对句子长度进行规范，因此这里的batch_size还是只能为1
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    """
        如果当前项目中，batch_size的值设置超过1，会报如下的错：
        RuntimeError: stack expects each tensor to be equal size, but got [10] at entry 0 and [6] at entry 1
    """
    # dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)

    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloader()

    for x, y in dataloader:
        print(f"英语句子-->{x.shape}-->{x}")
        print(f"法语句子-->{y.shape}-->{y}")
        break