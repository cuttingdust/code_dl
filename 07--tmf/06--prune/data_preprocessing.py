import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig

from tqdm import tqdm


from config import Config

# 1- 公共变量
config = Config()

bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)
bert_model = BertModel.from_pretrained(config.bert_path)
bert_config = BertConfig.from_pretrained(config.bert_path)


def load_raw_file(datapath):
    """
    加载并处理原始文件
    :param datapath: 原始文件路径
    :return: 处理后的文件，新闻标题string，目标值是int。格式：[(新闻标题,目标值),(新闻标题,目标值)...]
    """

    # 1- 读取原始文件内容
    with open(datapath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2- 循环遍历，处理每条样本
    result_list = []  # 返回结果
    for line in tqdm(lines, desc="loading data"):
        # 2.1- 空行处理和判断
        line = line.strip()
        if line == "":
            continue

        # 2.2- 每行数据拆解为新闻标题和目标值
        title, label = line.split("\t")

        # 【可选】健壮性代码
        """
            只要是有数据类型转换的地方，基本都有健壮性代码
        """
        if not label.isdigit():
            print(f"label的数据内容不合法，值是{label}")
            continue

        # 2.3- 存储到列表中
        result_list.append([title, int(label)])

    return result_list


if __name__ == "__main__":
    result_list = load_raw_file(config.dev_datapath)
    print(result_list[:10])
