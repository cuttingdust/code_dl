"""使用 PyTorch 的 ``nn.Linear`` 和 ``optim.SGD`` 完成一元线性回归。

本例要学习的数学关系是：

    y = wx + b + noise

其中：

    x       输入特征
    y       真实目标值
    w       斜率/权重，需要通过训练学习
    b       截距/偏置，需要通过训练学习
    noise   随机噪声，使数据更接近真实世界，不完全落在一条直线上

完整训练流程：

    生成数据
      ↓
    NumPy数组转换为Tensor
      ↓
    TensorDataset把x和y配对
      ↓
    DataLoader按批次提供数据
      ↓
    nn.Linear计算预测值
      ↓
    MSELoss计算预测值与真实值之间的误差
      ↓
    backward计算参数梯度
      ↓
    SGD根据梯度更新权重w和偏置b
      ↓
    重复多轮，直到损失逐渐降低
"""

import warnings

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_regression
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Matplotlib默认字体可能无法显示中文，因此指定中文字体并正常显示负号。
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def create_dataset() -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """生成一元线性回归训练数据，并转换为PyTorch张量。

    Returns:
        x: 输入特征，形状为 ``(100, 1)``。
        y: 真实目标值，形状为 ``(100,)``。
        true_weight: 生成数据时使用的真实斜率。
        true_bias: 生成数据时使用的真实截距。

    数据准备的一般流程是：

        文件/DataFrame/NumPy数组
            → Tensor
            → TensorDataset
            → DataLoader
    """

    # 真实截距由我们明确指定，所以程序知道它的值。
    # make_regression会随机生成真实斜率；coef=True让它把斜率一并返回。
    true_bias = 17.33

    x_numpy, y_numpy, coef_numpy = make_regression(
        n_samples=100,      # 生成100条样本
        n_features=1,       # 每条样本只有1个输入特征，因此是一元线性回归
        n_targets=1,        # 每条样本只有1个目标值
        bias=true_bias,     # 数据背后的真实截距b
        coef=True,          # 除x、y外，同时返回真实斜率w
        noise=10,           # 给y加入噪声，数据不会完美落在一条直线上
        shuffle=True,       # 打乱生成样本的排列顺序
        random_state=1024,  # 固定随机种子，使生成的数据能够复现
    )

    # make_regression返回NumPy数组：
    #
    #     x_numpy.shape = (100, 1)
    #     y_numpy.shape = (100,)
    #
    # PyTorch模型使用Tensor，因此将它们转换为float32张量。
    x = torch.tensor(x_numpy, dtype=torch.float32)
    y = torch.tensor(y_numpy, dtype=torch.float32)

    # n_features=1时coef_numpy只有一个元素。
    # .item()把单元素NumPy数组取成普通Python浮点数，便于显示和画真实直线。
    true_weight = float(coef_numpy.item())

    print("数据集准备完成：")
    print(f"x.shape          = {x.shape}")
    print(f"y.shape          = {y.shape}")
    print(f"真实斜率 w       = {true_weight:.6f}")
    print(f"真实截距 b       = {true_bias:.6f}")

    return x, y, true_weight, true_bias


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    true_weight: float,
    true_bias: float,
) -> nn.Linear:
    """训练线性回归模型、打印参数，并可视化损失和拟合结果。"""

    # ==================================================================
    # 第1步：使用TensorDataset和DataLoader组织训练数据
    # ==================================================================

    # TensorDataset按照相同下标把x和y配成一条样本：
    #
    #     dataset[0] = (x[0], y[0])
    #     dataset[1] = (x[1], y[1])
    #
    # 它不会复制数据，只是提供统一的数据集访问接口。
    dataset = TensorDataset(x, y)

    # DataLoader把100条数据拆成小批次提供给模型：
    #
    #     batch_size=16 → 大多数批次包含16条，最后一个批次包含4条
    #     shuffle=True  → 每一轮开始前重新打乱样本顺序
    #
    # 小批次训练的优点：
    # 1. 数据量很大时，不必一次把全部计算放进CPU/GPU内存。
    # 2. 相比逐条训练，矩阵批量运算效率更高。
    # 3. 批次带来的轻微随机性有时能帮助优化。
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ==================================================================
    # 第2步：定义线性模型、损失函数和优化器
    # ==================================================================

    # nn是Neural Network（神经网络）模块。
    # nn.Linear实现一个全连接线性层，数学公式为：
    #
    #     y_predict = x @ weight.T + bias
    #
    # in_features=1：每条样本输入1个特征。
    # out_features=1：每条样本输出1个预测值。
    # bias=True：模型同时学习截距。
    #
    # 模型内部参数形状：
    #
    #     model.weight.shape = (1, 1)
    #     model.bias.shape   = (1,)
    model = nn.Linear(in_features=1, out_features=1, bias=True)

    # MSELoss是均方误差损失：
    #
    #     loss = mean((y_predict - y_true)²)
    #
    # 预测越接近真实值，loss越小。默认reduction="mean"，因此返回标量。
    criterion = nn.MSELoss(reduction="mean")

    # optim是优化器模块，SGD是随机梯度下降优化器。
    #
    # model.parameters()把model.weight和model.bias交给优化器管理。
    # lr=0.01表示学习率，控制每次更新参数的步长。
    # momentum=0.9表示使用动量，让当前更新参考之前的更新方向：
    #
    #     无动量的基本思想：parameter -= lr × gradient
    #     使用动量：累计一部分历史梯度，使方向稳定并可能加快收敛
    #
    # 优化器只管理参数；只有调用optimizer.step()时才真正更新参数。
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("\n模型初始化完成：")
    print(model)
    print(f"初始 weight = {model.weight.item():.6f}")
    print(f"初始 bias   = {model.bias.item():.6f}")

    # ==================================================================
    # 第3步：训练模型
    # ==================================================================

    # epoch表示模型完整看完全部100条训练样本一次。
    # 100个epoch表示整个数据集被重复训练100轮。
    epochs = 100

    # 记录每个epoch的平均损失，用于绘制损失曲线。
    loss_list: list[float] = []

    for epoch in range(epochs):
        # 一个epoch由多个batch组成。
        # total_loss_value累计该epoch所有样本的损失总和。
        # total_sample_count累计该epoch实际处理的样本数。
        total_loss_value = 0.0
        total_sample_count = 0

        for batch_index, (x_train, y_train) in enumerate(dataloader):
            # x_train的典型形状是(16, 1)，表示16条样本、每条1个特征。
            # y_train原始形状是(16,)，但模型输出形状是(16, 1)。
            # 为了让MSELoss两边形状一致，使用reshape(-1, 1)把y调整为二维。
            y_train = y_train.reshape(-1, 1)

            # ----------------------------------------------------------
            # 3.1 梯度清零
            # ----------------------------------------------------------
            # PyTorch默认累加梯度。如果不清零，本批次梯度会与上一批次相加。
            # optimizer.zero_grad()会清除weight.grad和bias.grad。
            optimizer.zero_grad()

            # ----------------------------------------------------------
            # 3.2 前向传播
            # ----------------------------------------------------------
            # model(x_train)内部执行：
            #
            #     y_predict = x_train @ model.weight.T + model.bias
            y_predict = model(x_train)

            # 只在第一个epoch的第一个batch打印一次形状，避免每个batch都打印。
            if epoch == 0 and batch_index == 0:
                print("\n第一个训练批次的形状：")
                print(f"x_train.shape   = {x_train.shape}")
                print(f"y_train.shape   = {y_train.shape}")
                print(f"y_predict.shape = {y_predict.shape}")

            # ----------------------------------------------------------
            # 3.3 计算当前批次的平均损失
            # ----------------------------------------------------------
            loss_value = criterion(y_predict, y_train)

            # ----------------------------------------------------------
            # 3.4 反向传播
            # ----------------------------------------------------------
            # backward()通过计算图得到损失对weight和bias的梯度，并保存到：
            #
            #     model.weight.grad
            #     model.bias.grad
            #
            # MSELoss已经返回标量，所以可以直接调用backward()。
            # backward()只计算梯度，不会修改参数。
            loss_value.backward()

            # ----------------------------------------------------------
            # 3.5 更新参数
            # ----------------------------------------------------------
            # step()使用刚刚计算出的梯度和SGD规则更新weight、bias。
            # 这一步才是真正让模型学习的地方。
            optimizer.step()

            # ----------------------------------------------------------
            # 3.6 累计本epoch的损失
            # ----------------------------------------------------------
            # loss_value是当前batch的平均损失。
            # 最后一个batch只有4条样本，所以不能简单平均各batch的loss；否则
            # 4条样本的小batch会和16条样本的大batch拥有相同权重。
            #
            # 先乘当前batch样本数，恢复该batch的损失总和，最后再除总样本数。
            batch_size = len(x_train)
            total_loss_value += loss_value.item() * batch_size
            total_sample_count += batch_size

        # 全部batch训练完成，得到当前epoch在100条样本上的平均损失。
        average_loss = total_loss_value / total_sample_count
        loss_list.append(average_loss)

        # 每10轮打印一次；第一轮和最后一轮也打印，便于观察收敛趋势。
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"第 {epoch + 1:3d} 轮，"
                f"平均损失={average_loss:.6f}，"
                f"weight={model.weight.item():.6f}，"
                f"bias={model.bias.item():.6f}"
            )

    # ==================================================================
    # 第4步：查看训练后的参数
    # ==================================================================

    # state_dict()只保存模型参数，不保存模型类本身：
    #
    #     weight：训练得到的斜率
    #     bias：训练得到的截距
    print("\n训练完成：")
    print("model.state_dict() =", model.state_dict())
    print(f"真实斜率={true_weight:.6f}，学习斜率={model.weight.item():.6f}")
    print(f"真实截距={true_bias:.6f}，学习截距={model.bias.item():.6f}")

    # ==================================================================
    # 第5步：绘制epoch与损失的关系
    # ==================================================================

    # 如果训练正常，损失曲线通常会整体下降并逐渐稳定。
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_list)
    plt.xlabel("循环轮次 epoch")
    plt.ylabel("平均损失")
    plt.title("训练轮次与损失值的关系")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================================================================
    # 第6步：绘制训练数据、真实直线和模型预测直线
    # ==================================================================

    plt.figure(figsize=(8, 5))

    # x保存为(100,1)，绘制散点图时squeeze(1)将它变成(100,)。
    plt.scatter(x.squeeze(1), y, alpha=0.65, label="带噪声的训练样本")

    # 在x最小值与最大值之间生成1000个点，使直线显示得更加平滑。
    axis_x = torch.linspace(x.min(), x.max(), steps=1000)

    # 真实数据生成公式。这里必须使用生成数据时真实的bias=17.33。
    true_line = axis_x * true_weight + true_bias

    # 绘图属于模型推理，不需要建立计算图，因此使用torch.no_grad()。
    # nn.Linear要求输入形状为(N, 1)，所以先unsqueeze(1)增加特征维度；
    # 输出为(N, 1)，再用squeeze(1)变回(N,)以便绘图。
    model.eval()
    with torch.no_grad():
        predicted_line = model(axis_x.unsqueeze(1)).squeeze(1)

    plt.plot(axis_x, true_line, label="真实线性关系", color="red")
    plt.plot(axis_x, predicted_line, label="模型预测关系", color="blue")
    plt.xlabel("输入特征 x")
    plt.ylabel("目标值 y")
    plt.title("真实线性关系与模型预测结果")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    # 固定PyTorch随机种子，使nn.Linear的初始参数和DataLoader打乱顺序可复现。
    torch.manual_seed(42)

    # 1. 准备训练数据。
    features, targets, weight, bias = create_dataset()

    # 2. 训练模型并展示结果。
    trained_model = train_model(features, targets, weight, bias)
