import torch

def demo01() -> None:
    # 真实值：真实值可以是分类的ID，也可以是分类ID热编码之后的值
    # 分类ID热编码之后
    # y_true = torch.tensor([[0,1,0],[0,0,1]],dtype=torch.float32)

    # 分类的ID，推荐用这种。注意：分类的ID数据类型必须是long
    y_true = torch.tensor([1, 2], dtype=torch.long)

    # 预测值：也就是线性求和的结果
    y_pred = torch.tensor([[2, 2, 4], [3, 1, 6]], requires_grad=True, dtype=torch.float32)

    # 交叉熵损失对象
    loss = torch.nn.CrossEntropyLoss()

    #计算损失值
    loss_value = loss(y_pred, y_true)
    print(f"损失值是{loss_value}")


def demo02():
    # 真实概率值，要么是0要么是1
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)

    # 预测概率值，取值范围[0,1]
    # [0.6901, 0.5459, 0.2469]代表3条样本的预测为某一种类别的概率值
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469],requires_grad=True,dtype=torch.float32)

    # 二分类损失对象
    """
        计算过程：
            第1条样本的：-0*loge0.6901-(1-0)*loge(1-0.6901)
            第2条样本的：-1*loge0.5459-(1-1)*loge(1-0.5459)
            第3条样本的：-0*loge0.2469-(1-0)*loge(1-0.2469)

            累加求和，再除以样本条数
    """
    loss = torch.nn.BCELoss()

    # 计算损失值
    loss_value = loss(y_pred, y_true)
    print(f"二分类损失值：{loss_value}")


if __name__ == '__main__':
    # 多分类损失函数
    # demo01()

    # 二分类损失函数
    demo02()