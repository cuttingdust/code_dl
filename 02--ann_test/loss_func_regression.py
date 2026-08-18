import torch

"""
    回归的损失函数总结
        1- L1损失函数，也是MAE
            缺点：在0处不可导，会导致错过损失值最小的位置
            使用：偶尔使用

        2- L2损失函数，也是MSE
            缺点：因为公式中对损失值计算了平方，会导致出现梯度爆炸
            使用：很少使用

        3- SmoothL1Loss
            特点：摒弃掉了L1和L2损失函数的确定。
            使用：推荐使用
"""


def mae() -> None:
    # 真实值
    y_true = torch.tensor([2, 2, 2])

    # 预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9],requires_grad=True,dtype=torch.float32)

    # 创建损失函数对象
    loss = torch.nn.L1Loss()

    # 计算损失值
    loss_value = loss(y_pred, y_true)
    print(f"MAE损失值为{loss_value}")


def mse() -> None:
    # 真实值
    y_true = torch.tensor([2, 2, 2])

    # 预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True, dtype=torch.float32)

    # 创建损失函数对象
    loss = torch.nn.MSELoss()

    # 计算损失值
    loss_value = loss(y_pred, y_true)
    print(f"MSE损失值为{loss_value}")


def smooth_l1() -> None:
    # 真实值
    y_true = torch.tensor([2, 2, 2])

    # 预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True, dtype=torch.float32)

    # 损失函数对象
    loss = torch.nn.SmoothL1Loss()

    # 损失值
    loss_value = loss(y_pred, y_true)
    print(f"smooth_l1损失值为{loss_value}")

if __name__ == '__main__':
    # MAE-->L1 Loss
    mae()

    # MSE-->L2 Loss
    mse()

    # smooth_l1
    smooth_l1()