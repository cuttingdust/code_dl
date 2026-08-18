"""
    搭建神经网络的步骤：
        1- 定义一个类，继承自torch.nn.Module
        2- 实现两个方法：
            2.1- __init__：初始化方法。负责的工作内容如下：  --》定义网络结构，也就是隐藏层和输出层的层数、每层的神经元个数w、b参数初始化
                a、初始化父类
                b、设置属性值
                c、定义神经网络结构：隐藏层有几层、输出层是怎样的等
                d、参数初始化：w、b如何进行初始化

            2.2- forward：前向传播。 输入样本数据，经过各个网络层，得到预测结果
"""
import torch
import torch.nn as nn
# 安装命令 pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/
from torchsummary import summary


class MyModel(nn.Module):
    def __init__(self):
        # a、初始化父类：也就是调用父类中的初始化方法
        super(MyModel,self).__init__()

        # b、设置属性值【目前不需要】

        # c、定义神经网络结构：隐藏层有几层、输出层是怎样的等
        # 1.1- 隐藏层
        # 第一层隐藏层
        """
            参数解释：
                in_features：输入的特征个数。也就是上游神经元的个数
                out_features：输出的特征个数。也就是本层神经元的个数
        """
        self.linear1 = nn.Linear(in_features=3,out_features=3)

        # 第二层隐藏层
        self.linear2 = nn.Linear(in_features=3, out_features=2)

        # 1.2- 输出层
        self.output = nn.Linear(in_features=2, out_features=2)

        # d、参数初始化：w、b如何进行初始化
        # 隐藏层1的初始化
        nn.init.xavier_normal_(self.linear1.weight)  # w初始化
        nn.init.zeros_(self.linear1.bias)  # b初始化

        # 隐藏层2的初始化
        nn.init.kaiming_normal(self.linear2.weight)  # w初始化
        nn.init.zeros_(self.linear2.bias) # b初始化

    def forward(self, data):
        """
        前向传播。对模型进行训练，也就是得到预测结果
        :param data: 前向传播送入到网络结构中的样本数据。类型是张量tensor
        :return:
        """
        # 1- 数据经过第一层隐藏层
        # 1.1- 分开版
        # 先进行线性求和
        # linear1_result = self.linear1(data)
        # 再将线性求和结果给到激活函数，得到激活值
        # data = torch.sigmoid(linear1_result)

        # 1.2- 合并版【推荐】
        data = torch.sigmoid(self.linear1(data))

        # 2- 数据经过第二层隐藏层
        data = torch.relu(self.linear2(data))

        # 3- 数据经过输出层得到输出结果
        """
            dim=-1：不管列表嵌套多少层，全部只对最里层（最后一个维度）的数据进行统计
        """
        return torch.softmax(self.output(data),dim=-1)


def train_model():
    # 1- 准备训练数据：目前的需求中，每条样本必须有3个特征
    # 10：多少条样本；3：每条样本多少个特征
    data = torch.randn(10, 3)

    # 2- 创建神经网络模型对象
    my_model = MyModel()

    # 3- 模型训练：将数据输入到神经网络中，进行前向传播
    # 内部自动调用forward方法
    output = my_model(data)
    print(f"预测结果是：{output}")
    print(f"预测结果形状：{output.shape}")

    print("-" * 30)

    # 4- 查看神经网络的概要信息
    """
        summary(参数1,参数2,参数3)：查看神经网络各层参数的信息
            参数1：神经网络算法实例对象
            参数2：输入的特征个数，类型必须是元组
            参数3：输入的每个批次样本条数，任意
    """
    summary(my_model,input_size = (3,), batch_size=2, device="cpu")

    print("-"*30)


if __name__ == "__main__":
    train_model()