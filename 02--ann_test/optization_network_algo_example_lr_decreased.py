import torch
from matplotlib import pyplot as plt
from torch import optim

def demo01() -> None:
    # 定义相关数据：样本真实值、特征值、初始化权重
    y_true = torch.tensor([0])
    x = torch.tensor([1], dtype=torch.float32)
    w = torch.tensor([1], requires_grad=True ,dtype=torch.float32)

    # 梯度下降优化器
    optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)

    # 创建等间隔学习率衰减策略
    """
        optimizer：梯度下降优化器
        step_size：间隔多少轮次对学习率进行衰减
        gamma：衰减的时候，上一次的学习率乘以的系数是多少。更新后的学习率 = 上一次的学习率 * gamma
    """
    # 掌握
    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=50, gamma=0.5)

    # 训练相关的次数
    epochs = 200
    batch_size = 8
    epoch_list = [] # 记录轮次信息
    lr_list = []  # 记录学习率的值

    for epoch in range(epochs): # 轮次
        for i in range(batch_size): # 批次
            # 获得预测值
            y_pred = w * x

            # 计算损失值
            loss_value = (y_pred - y_true)**2

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss_value.sum().backward()
            # 更新权重
            optimizer.step()

        # 记录轮次和学习率信息
        epoch_list.append(epoch)
        # scheduler.get_last_lr()这个获取学习率的代码要在scheduler.step()之前获得
        lr_list.append(scheduler.get_last_lr())  # 0.1 0.1 0.1 0.1 0.1  .... 0.05 0.05 0.05 ..... 0.025 0.025
        print(f"第{epoch + 1}轮次，学习率是{scheduler.get_last_lr()}")

        # 按照等间隔更新学习率
        # 掌握
        scheduler.step()

    # 绘制学习率变化曲线
    plt.plot(epoch_list, lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()


def demo02() -> None:
    # 定义相关数据：样本真实值、特征值、初始化权重
    y_true = torch.tensor([0])
    x = torch.tensor([1], dtype=torch.float32)
    w = torch.tensor(1, requires_grad=True ,dtype=torch.float32)

    # 梯度下降优化器
    optimizer = optim.SGD([w], lr=0.1, momentum=0.9)

    # 创建指定间隔学习率衰减策略
    # 掌握
    epoch_milestones = [50, 125, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=epoch_milestones, gamma=0.5)

    # 训练相关的次数
    epochs = 200
    batch_size = 8
    epoch_list = []  # 记录轮次信息
    lr_list = []  # 记录学习率的值

    for epoch in range(epochs):
        for i in range(batch_size):
            # 获得预测值
            y_pred = w * x

            # 计算损失值
            loss_value = (y_pred - y_true)**2

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss_value.sum().backward()
            # 更新权重
            optimizer.step()

        # 记录轮次和学习率信息
        epoch_list.append(epoch)
        # scheduler.get_last_lr()这个获取学习率的代码要在scheduler.step()之前获得
        lr_list.append(scheduler.get_last_lr())
        print(f"第{epoch + 1}轮次，学习率是{scheduler.get_last_lr()}")

        # 按照指定间隔更新学习率
        # 掌握
        scheduler.step()

        # 绘制学习率变化曲线
    plt.plot(epoch_list, lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()


def demo03() -> None:
    # 定义相关数据：样本真实值、特征值、初始化权重
    y_true = torch.tensor([0])
    x = torch.tensor([1], dtype=torch.float32)
    w = torch.tensor(1, requires_grad=True, dtype=torch.float32)

    # 梯度下降优化器
    optimizer = optim.SGD([w], lr=0.1, momentum=0.9)

    # 创建学习率指数衰减策略
    # 掌握
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # 训练相关的次数
    epochs = 200
    batch_size = 8
    epoch_list = []  # 记录轮次信息
    lr_list = []  # 记录学习率的值

    for epoch in range(epochs):
        for i in range(batch_size):
            # 获得预测值
            y_pred = w * x

           # 计算损失值
            loss_value = (y_pred-y_true)**2

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss_value.sum().backward()
            # 更新权重
            optimizer.step()

        # 记录轮次和学习率信息
        epoch_list.append(epoch)
        # scheduler.get_last_lr()这个获取学习率的代码要在scheduler.step()之前获得
        lr_list.append(scheduler.get_last_lr())
        print(f"第{epoch + 1}轮次，学习率是{scheduler.get_last_lr()}")

        # 按照学习率指数衰减策略
        # 掌握
        scheduler.step()


    # 绘制学习率变化曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

if __name__ == '__main__':
    # 等间隔学习率
    # demo01()

    # 指定间隔学习率
    # demo02()

    # 学习率指数衰减策略
    demo03()