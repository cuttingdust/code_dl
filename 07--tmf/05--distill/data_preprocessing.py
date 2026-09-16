import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

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
    with open(datapath, mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2- 循环遍历，处理每条样本
    result_list = []  # 返回结果
    for line in tqdm(lines, desc="处理文件中..."):
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
        result_list.append((title, int(label)))

    return result_list


# 自定义数据集
class NewsDataset(Dataset):
    def __init__(self, data_set):
        self.data_list = data_set

        # 样本条数
        self.sample_len = len(self.data_list)

    def __len__(self):
        # 获得样本条数
        return self.sample_len

    def __getitem__(self, index):
        # 防止index出现负数和越界
        index = min(max(index, 0), self.sample_len - 1)

        # 获得数据
        title, label = self.data_list[index]

        return title, label


# 对每个批次的数据进行特定的处理
def collate_fn(batch_data):
    # 1- 将每个批次中新闻标题组织成一个容器；label目标值也单独组织成一个容器
    """
    zip(*)处理过程如下：
        输入数据：[('近期新盘推荐 通州纯新别墅本周开盘', 1), ('陕西退休教师嫌弃精神病女儿将其勒死被捕', 5)]
        输出数据：[('近期新盘推荐 通州纯新别墅本周开盘', '陕西退休教师嫌弃精神病女儿将其勒死被捕'),     (1, 5)]
    """
    titles, labels = zip(*batch_data)
    # print(list(zip(*batch_data)))
    # print(type(batch_data))
    # print(batch_data)

    # 2- 对新闻标题进行分词处理，得到词索引张量
    title_tensor = bert_tokenizer(
        titles,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
    )
    # print(title_tensor)
    # print(type(title_tensor))

    # 3- 解析分词后的结果
    input_ids = title_tensor.input_ids
    attention_mask = title_tensor.attention_mask
    # token_type_ids表示每个词来自第几段文本。当前是单句分类，所以值全是0。
    # 教师模型forward仍然接收该参数，因此这里一并返回，确保与训练循环的4个变量对应。
    token_type_ids = title_tensor.token_type_ids

    # 4- 转成张量并返回
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_mask, token_type_ids, labels


# 自定义数据加载器
def build_dataloader(datapath, shuffle=False):
    # 1- 加载原始文件
    data_list = load_raw_file(datapath)

    # 2- 创建自定义数据集
    dataset = NewsDataset(data_list)

    # 3- 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    print("")
    # result_list = load_raw_file(config.dev_datapath)
    # print(result_list[:10])

    for input_ids, attention_mask, labels in build_dataloader(
        config.train_datapath, True
    ):
        print("input_ids-->", input_ids)
        print("attention_mask-->", attention_mask)
        print("labels-->", labels)
        break
