import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset  # HuggingFace数据加载方法
from tqdm import tqdm

# 1- 定义变量
# 运行的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 预训练模型路径
model_path = r"PretrainedModel/bert-base-chinese"
# 创建词汇映射器：词汇映射器只是对句子进行分词，然后得到词索引，不涉及任何的需要学习的参数，因此可以不用放到GPU或CPU上
tokenizer = BertTokenizer.from_pretrained(model_path)
# 创建Bert预训练模型实例对象：需要学习相关的模型参数，因此必须放到GPU或CPU上
bert_model = BertModel.from_pretrained(model_path).to(device=device)


# 2- 数据探索【可选】
def load_data():
    """
    参数解释：
        path：既能够指定文件目录，也能够指定文件类型。推荐用来指定文件目录
        data_files：可以传递文件路径，还可以传递数据类型和文件路径字典的形式。推荐直接传递文件路径
        split：如果data_files是具体的文件路径，那么这里必须给train
    """
    # 加载方式一：【掌握】如果data_files是具体的文件路径，那么这里必须给train
    data = load_dataset(path="data", data_files="train.csv", split="train")

    # 加载方式二：如果data_files是具体的文件路径，那么这里必须给train
    # data = load_dataset(path="data", data_files="test.csv", split="train")

    # 加载方式三：path：既能够指定文件目录，也能够指定文件类型。推荐用来指定文件目录
    # data = load_dataset(path="csv", data_files="data/test.csv", split="train")

    # 加载方式四：数据类型和文件路径以字典的形式传递
    # file_dict = {
    #     "train":"data/train.csv",
    #     "test":"data/test.csv",
    #     "aaaa":"data/test.csv"
    # }
    # data = load_dataset(path="csv", data_files=file_dict, split="train")
    # data = load_dataset(path="csv", data_files=file_dict, split="test")
    # data = load_dataset(path="csv", data_files=file_dict, split="aaaa")

    print(type(data))
    print(len(data))
    print(data)
    print(data[0])


# 3- 针对每个批次数据的具体处理函数
def collate_fn(batch_data):
    # 1- 分别提取出特征和目标值
    # 数据示例：[句子1, 句子2, ...]
    sents = [line_dict.get("text") for line_dict in batch_data]

    # 数据示例：[1, 0, 1, 1, 1, 0, 1, 0]
    labels = [line_dict.get("label") for line_dict in batch_data]

    # 2- 数据预处理
    # 2.1- 特征进行分词，得到词索引组成的张量
    # 因为DataLoader中batch_size>1，因此这里的padding必须为max_length，不能为True
    sents_tensor = tokenizer(
        sents,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=300,
    )
    # print(f"sents_tensor = {sents_tensor}")

    # 将sents_tensor字典拆解出来
    input_ids = sents_tensor["input_ids"]
    token_type_ids = sents_tensor["token_type_ids"]
    attention_mask = sents_tensor["attention_mask"]

    # print(f"input_ids = {input_ids}")
    # print(f"token_type_ids = {token_type_ids}")
    # print(f"attention_mask = {attention_mask}")

    # 2.3- 目标值直接封装为张量
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, token_type_ids, attention_mask, labels


# 4- 创建数据加载器
def get_dataloader(file_path):
    # 1- 加载指定路径下的文件
    dataset = load_dataset(path="data", data_files=file_path, split="train")

    # 2- 创建DataLoader实例对象
    """
    参数解释：
        dataset：数据集对象
        batch_size：每个批次中样本的条数。如果样本长度不一致，batch_size只能设置为1
        shuffle：是否对数据打散。让模型训练更加充分
        drop_last：如果最后一个批次的样本条数不足batch_size大小，那么直接不要
        collate_fn：针对每个批次数据的具体处理函数。注意：传递的是函数名称，不要带小括号
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader


# 5- 搭建模型：Bert预训练模型+我们自定义的网络层
class AiModel(nn.Module):
    def __init__(self):
        # 1- 初始化父类
        super().__init__()

        # 2- 在Bert预训练模型的后面，增加自己的线性网络结构层
        """
            这里的参数必须是768和2，为什么？
            答：
                768：Bert的词向量/隐藏状态向量维度是768维
                2：目前案例是二分类场景
        """
        self.linear = nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 1- 先调用Bert预训练模型
        """
        注意如下两点：
            1- 推荐使用torch.no_grad()，冻结Bert的参数训练。可以不加，那么回对Bert的110M个参数都会进行训练，比较耗时
            2- bert_model()里面的参数要使用关键字传参
        """
        with torch.no_grad():
            bert_output = bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        # 2- 再调用我们自定义的网络层
        """
            为什么写成bert_output.last_hidden_state[:, 0]？
            答：
                1- last_hidden_state表示取的是Bert模型最后一层隐藏层中每个时间步的隐藏状态。等价于多层RNN中output
                2- [:, 0]取到的是每条句子中第一个token（词），也就是[CLS]的隐藏状态
                3- Bert预训练模型的设计思想：只有编码器端，编码器端注意力是自注意力，也就是每个词都能够获得其他词的隐藏状态信息
                    [CLS]是一个固定的标识，每个句子的开头都有
                4- 结论：Bert做分类，固定写成bert_output.last_hidden_state[:, 0]

                以上的解释，建议看论文：https://arxiv.org/pdf/1810.04805
                对应内容：The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
        """
        # bert_out.last_hidden_state的形状：
        # [batch_size, sequence_length, hidden_size]
        # 例如：[8, 300, 768]
        #
        # [:, 0, :] 的含义：
        # 第1个冒号：取当前批次中的所有句子；
        # 0：取每条句子的第0个token，也就是[CLS]；
        # 第2个冒号：取[CLS]对应的全部768维特征。
        # 得到的cls_hidden_state形状为：[batch_size, 768]。
        cls_hidden_state = bert_output.last_hidden_state[:, 0, :]

        # 线性层把每条句子的768维综合特征转换成2个分类分数（logits）。
        # 输出形状：[batch_size, 2]，例如：[8, 2]。
        # 这里不要手动调用softmax：训练时CrossEntropyLoss内部会完成相应计算。
        logits = self.linear(cls_hidden_state)

        return logits


# 6- 模型训练
def train():
    # 1- 加载数据
    dataloader = get_dataloader("train.csv")

    # 2- 禁用Bert预训练模型的梯度下降
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 3- 创建对象
    # 3.1- 创建模型对象
    model = AiModel().to(device)
    # 3.2- 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    # 3.3- 损失函数
    loss = nn.CrossEntropyLoss()

    # 4- 切换模式
    model.train()

    # 5- 训练
    epochs = 1
    for epoch in range(epochs):

        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(
            tqdm(dataloader), start=1
        ):
            # 5.1- 将数据全部发送到对应的设备
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 5.2- 前向传播
            pred_result = model(input_ids, token_type_ids, attention_mask)

            # 5.3- 计算损失值
            loss_value = loss(pred_result, labels)

            # 5.4- 固定代码
            optimizer.zero_grad()  # 梯度清零
            loss_value.sum().backward()  # 反向传播
            optimizer.step()  # 更新参数：不更新Bert的，只更新我们自己增加的线性层

            # 6- 间隔输出相关指标
            # 6- 间隔输出相关指标
            if i % 20 == 0:
                # 6.1- 获得预测概率最高的索引
                pred_index = torch.argmax(pred_result, dim=-1)
                # 6.2- 统计预测正确的样本条数
                correct_count = (pred_index == labels).sum().item()
                # 6.3- 计算准确率
                acc = correct_count / len(labels)
                print(f"第{epoch+1}轮次，已训练的样本批次{i}，平均准确率{round(acc,4)}")

    # 7- 保存训练好的模型
    torch.save(model.state_dict(), "model/bert.pkl")


# 7- 模型预测
def predict():
    # 1- 加载数据
    dataloader = get_dataloader("test.csv")
    # 2- 加载训练好的模型
    model = AiModel().to(device)
    model.load_state_dict(torch.load("model/bert.pkl"))

    # 3- 定义准确率统计变量
    correct_count = 0  # 预测正确的样本条数
    total_sample_count = 0  # 已经预测的总样本条数

    # 4- 预测
    model.eval()
    with torch.no_grad():

        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(
            tqdm(dataloader), start=1
        ):
            # 4.1- 将数据发送到对应设备
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 4.2- 预测得到结果
            pred_result = model(input_ids, token_type_ids, attention_mask)

            # 4.3- 计算准确率：算的累计平均准确率
            pred_index = torch.argmax(pred_result, dim=-1)
            correct_count += (pred_index == labels).sum().item()
            total_sample_count += len(labels)

            # print("labels-->",labels)
            # print("pred_index-->",pred_index)
            # print("input_ids-->",input_ids)

            # 4.4- 间隔输出相关指标
            if i % 20 == 0:
                # 计算准确率
                acc = correct_count / total_sample_count
                print(f"已预测的样本批次{i}，累计平均准确率{round(acc, 4)}")

                # 恢复得到原始句子内容
                text_content = "".join(
                    tokenizer.convert_ids_to_tokens(
                        input_ids[0], skip_special_tokens=True
                    )
                )
                print(
                    f"每个批次中第一条样本的原始评价内容：{text_content}，预测的类别是：{pred_index[0]}"
                )


if __name__ == "__main__":
    # 测试数据探索
    # load_data()

    # 测试数据加载器
    # dataloader = get_dataloader(file_path="train.csv")
    # for input_ids, token_type_ids, attention_mask, labels in dataloader:
    #     print(token_type_ids)
    #     break

    # train()

    predict()
