import torch
from redis.commands.search import result
from  torch import nn

if __name__ == '__main__':
    # 准备输入的样本数据
    inputs = torch.randint(low=1, high=10, size=(1,4), dtype=torch.float32)
    print(f"输入数据内容{inputs}")

    # 构建神经网络结构
    hidden = nn.Linear(4, 5)
    dropout = nn.Dropout(p=0.4)

    # 计算数据
    result = hidden(inputs) # 隐藏层进行线性求和
    result = torch.tanh(result)  # 隐藏层进激活函数
    print(f"激活函数后结果{result}")
    result = dropout(result)  # Dropout层随机失活。将Dropout当成一层神经网络层使用即可
    print(f"随机失活后结果{result}")

