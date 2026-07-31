import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# 1- 数据处理：文件->DataFrame->Tensor张量->Dataset->DataLoader
def create_dataset():
    # 1- 读取文件
    df = pd.read_csv(r"data/手机价格预测.csv", encoding="utf-8")

    # 2- 拆分得到特征数据和目标值
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3- 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=201
    )

    # TODO 特征标准化处理

    # 4- DataFrame->Tensor张量
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    # values：只取数据内容，字段名称等不需要
    # int64是long类型
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    y_test = torch.tensor(y_test.values, dtype=torch.int64)

    # 5- Tensor张量->Dataset
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # 6- 其他数据信息
    # 6.1- 获得特征个数
    feature_nums = x.shape[1]

    # 6.2- 获得目标值个数：去重
    target_nums = len(np.unique(y))

    return train_dataset, test_dataset, feature_nums, target_nums


# 2- 自定义网络结构
class PhonePriceModel(nn.Module):
    def __init__(self, feature_nums, target_nums):
        # 1- 初始化父类
        super(PhonePriceModel, self).__init__()

        # 2- 设置属性值
        self.feature_nums = feature_nums
        self.target_nums = target_nums

        # 3- 搭建网络结构
        # 3.1- 第一层隐藏层
        # 注意：in_features需要和数据中特征个数完全一致
        self.linear1 = nn.Linear(in_features=self.feature_nums, out_features=256)
        # 3.2- 第二层隐藏层
        self.linear2 = nn.Linear(in_features=256, out_features=512)
        # 3.3- 第三层隐藏层
        self.linear3 = nn.Linear(in_features=512, out_features=256)
        # 3.4-输出层
        self.output = nn.Linear(in_features=256, out_features=self.target_nums)

    def forward(self, data):
        # 1- 经过第一层隐藏层
        data = torch.relu(self.linear1(data))

        # 2- 经过第二层隐藏层
        data = torch.relu(self.linear2(data))

        # 3- 经过第三层隐藏层
        data = torch.relu(self.linear3(data))

        # 4- 经过输出层得到结果
        # 为什么这里不明确的调用softmax激活函数？因为后续训练的时候会调用CrossEntropyLoss，里面自带softmax
        return self.output(data)


# 3- 模型训练
def train_model(train_dataset, test_dataset, feature_nums, target_nums):
    # 1- 得到数据加载器，Dataset->DataLoader：为了防止内存溢出
    torch.manual_seed(201)  # 设置随机数种子，让数据的划分固定下来
    """
      参数解释：
            dataset：数据集合
            batch_size：同一时刻送入到神经网络中的样本条数。训练效果会更好，提高模型的处理吞吐能力
            shuffle：对数据打散
    """
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    # 2- 创建网络模型实例对象
    model = PhonePriceModel(feature_nums, target_nums)

    # 3- 创建损失函数对象：多分类问题需要使用交叉熵损失函数
    loss = nn.CrossEntropyLoss()

    # 4- 创建优化器对象
    """
        SGD：随机梯度下降算法。每次随机选一条样本计算梯度值
        params：告诉梯度下降算法，要帮我对什么参数进行优化（梯度下降），这里就是w和b
        lr：学习率
    """
    optimzer = optim.SGD(params=model.parameters(), lr=1e-3)

    # 5- 循环训练
    epochs = 50  # 对总样本，总共训练多少个轮次
    model.train()  # 切换模式为训练模式。也就是允许神经网络随机失活

    """
        总样本条数 100
        每个批次中有8条样本
        那么：每个轮次中有13个批次
    """
    for epoch in range(epochs):
        # 外层循环控制轮次

        # 【可选】记录损失值的变化过程：也就是计算平均损失 = total_loss_value / total_sample_num
        total_loss_value = 0.0  # 每个轮次中总的损失值
        total_sample_num = 0  # 每个轮次中已经训练的样本条数

        for x, y in dataloader:
            # x样本中的特殊数据；y样本中的真实值
            # 内层循环控制批次

            # 5.1- 前向传播
            pred_y = model(x)

            # 5.2- 计算损失值
            loss_value = loss(pred_y, y)

            # 【可选】更新损失值
            # item()将张量变成标量
            total_loss_value += loss_value.item()
            total_sample_num += len(x)

            # 5.3- 固定代码
            # 梯度清零：默认会对梯度值进行累积
            optimzer.zero_grad()
            # 反向传播：注意只有标量张量能进行反向传播
            loss_value.sum().backward()
            # 更新参数：也就是更新w和b
            optimzer.step()

        print(f"第{epoch + 1}次，总的平均损失是：{total_loss_value / total_sample_num}")

    model_dir = Path(__file__).parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 6- 保存训练好的模型
    torch.save(model.state_dict(), "model/phone_price_model.pkl")


# 4- 模型预测：代码层面就是简化版的模型训练代码
def predict_model(test_dataset, feature_nums, target_nums):
    # 1- 创建数据加载器
    # 注意：模型预测的时候shuffle要设置为False，也就是不让数据打散，为了让预测结果稳定
    dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    # 2- 创建算法模型
    model = PhonePriceModel(feature_nums, target_nums)

    # 3- 加载训练好的模型
    model.load_state_dict(torch.load("model/phone_price_model.pkl"))

    # 4- 预测
    model.eval()  # 切换为预测模式。禁止随机失活Dropout
    correct_cnt = 0  # 预测正确的总样本条数

    for x, y in dataloader:
        # 4.1- 前向传播：预测
        pred_y = model(x)

        """
            1- pred_y：只是输出层中线性求和的结果，并没有经过softmax的处理
            2- 可以手动调用softmax，将线性结果处理成概率值。torch.softmax(pred_y,dim=-1)
        """
        print(f"pred_y-->{pred_y}")
        print(f"pred_y经过softmax的处理-->{torch.softmax(pred_y,dim=-1)}")

        # 4.2- 获得预测概率值最高的索引。实际就是获得了预测目标值的类别
        pred_id = torch.argmax(pred_y, dim=-1)
        print(f"argmax的结果-->{pred_id}")

        # 4.3- 统计预测正确的数据条数
        print(f"真实值-->{y}")
        print(f"判断结果-->{(pred_id==y)}")
        correct_cnt = correct_cnt + (pred_id == y).sum()

    # 5- 计算准确率
    acc_rate = correct_cnt / len(test_dataset)
    print(
        f"预测的准确率是：{acc_rate}，预测正确的样本条数：{correct_cnt}，预测的总样本条数：{len(test_dataset)}"
    )


if __name__ == "__main__":
    train_dataset, test_dataset, feature_nums, target_nums = create_dataset()
    print(train_dataset)
    print("-" * 50)
    print(test_dataset)
    print("-" * 50)
    print(feature_nums)
    print("-" * 50)
    print(target_nums)

    for i in train_dataset:
        print(i)
        break

    # train_model(train_dataset, test_dataset, feature_nums, target_nums)

    predict_model(test_dataset, feature_nums, target_nums)
