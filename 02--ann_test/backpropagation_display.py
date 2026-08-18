"""展示一个线性模型从前向传播到反向传播的完整过程。

模型公式：

    Z = XW + b

其中：

    X：输入样本，形状为 (样本数, 输入特征数)
    W：权重矩阵，形状为 (输入特征数, 输出特征数)
    b：偏置，形状为 (输出特征数,)
    Z：模型预测，形状为 (样本数, 输出特征数)

本例使用均方误差 MSE 衡量预测值 Z 与真实值 Y 的差距：

    loss = mean((Z - Y)²)

最后调用 backward()，让 PyTorch 根据链式法则自动计算：

    ∂loss/∂W，保存在 w.grad
    ∂loss/∂b，保存在 b.grad

注意：backward() 只计算梯度，不会自动修改 w 和 b。真正更新参数还需要
手动执行梯度下降，或者使用 torch.optim 中的优化器。
"""

import torch


if __name__ == "__main__":
    # 固定随机种子，使每次运行产生相同的随机权重和偏置，方便学习和调试。
    # 如果不设置随机种子，torch.randn() 每次通常会生成不同的随机数，
    # 因而每次运行得到的预测值、损失值和梯度也会不同。
    torch.manual_seed(42)

    # ==================================================================
    # 第一部分：准备输入数据和真实值
    # ==================================================================

    # x 是模型输入：2 条样本，每条样本包含 5 个输入特征。
    #
    # x.shape = (2, 5)
    #
    # 当前所有元素都是 1：
    #
    #     x = [[1, 1, 1, 1, 1],
    #          [1, 1, 1, 1, 1]]
    #
    # 两条样本完全相同，所以使用同一组 w 和 b 时，得到的预测也会相同。
    # x 是训练数据而不是待学习参数，因此默认 requires_grad=False。
    x = torch.ones(2, 5, dtype=torch.float32)

    # y 是两条样本对应的真实目标值。
    # 每条样本需要预测 3 个数，因此 y.shape=(2, 3)。
    # 当前真实值全部设置为 0，只是为了简化演示。
    y = torch.zeros(2, 3, dtype=torch.float32)

    print("输入 x：")
    print(x)
    print(f"x.shape = {x.shape}，x.requires_grad = {x.requires_grad}")
    print("\n真实值 y：")
    print(y)
    print(f"y.shape = {y.shape}，y.requires_grad = {y.requires_grad}")

    # ==================================================================
    # 第二部分：初始化可训练参数 w 和 b
    # ==================================================================

    # w 是权重矩阵。
    # 输入有 5 个特征、输出有 3 个特征，所以 w.shape 必须是 (5, 3)。
    # torch.randn() 从均值为 0、标准差为 1 的标准正态分布中生成随机数。
    # requires_grad=True 表示需要跟踪 w 参与的运算，以便计算 loss 对 w 的梯度。
    w = torch.randn(5, 3, dtype=torch.float32, requires_grad=True)

    # b 是偏置，每个输出对应一个偏置，所以 b.shape=(3,)。
    # b 也是需要训练的参数，因此同样设置 requires_grad=True。
    b = torch.randn(3, dtype=torch.float32, requires_grad=True)

    # w 和 b 都是直接创建的叶子张量。反向传播完成后，它们的梯度会分别
    # 保存在 w.grad 和 b.grad 中。此时还没有 backward()，所以二者均为 None。
    print("\n初始化权重 w：")
    print(w)
    print(f"w.shape = {w.shape}，反向传播前 w.grad = {w.grad}")
    print("\n初始化偏置 b：")
    print(b)
    print(f"b.shape = {b.shape}，反向传播前 b.grad = {b.grad}")

    # ==================================================================
    # 第三部分：前向传播，计算预测值 z = x @ w + b
    # ==================================================================

    # @ 是 Python 的矩阵乘法运算符，相当于：
    #
    #     torch.matmul(x, w)
    #     x.matmul(w)
    #
    # 矩阵形状变化为：
    #
    #     x          @ w          = x @ w
    #     (2, 5)       (5, 3)       (2, 3)
    #
    # 矩阵乘法要求左矩阵的列数等于右矩阵的行数，这里中间的 5 相等；
    # 运算结果保留外侧两个维度，因此得到 (2, 3)。
    #
    # b.shape=(3,)，PyTorch 使用广播机制，把同一个 b 加到两条样本上：
    #
    #     [[z11, z12, z13],      [[b1, b2, b3],
    #      [z21, z22, z23]]  +    [b1, b2, b3]]
    #
    # 这里没有真的复制 b，只是按照广播规则完成运算。
    z = x @ w + b

    # z 是由 x、w、b 计算出来的预测值，也是非叶子张量。
    # 因为 w、b 需要梯度，所以 z.requires_grad=True，并带有 grad_fn。
    print("\n模型预测值 z = x @ w + b：")
    print(z)
    print(f"z.shape = {z.shape}")
    print(f"z.requires_grad = {z.requires_grad}")
    print(f"z.grad_fn = {z.grad_fn}")

    # ==================================================================
    # 第四部分：定义并计算损失
    # ==================================================================

    # torch.nn 是 PyTorch 提供神经网络组件的模块，nn 是 Neural Network。
    # MSELoss 是均方误差损失函数，MSE 表示 Mean Squared Error。
    #
    # 默认 reduction="mean"，计算规则为：
    #
    #     loss = 所有 (预测值 - 真实值)² 的平均值
    #
    # loss_fn 是“损失函数对象/计算规则”，还不是本次计算的损失值。
    loss_fn = torch.nn.MSELoss(reduction="mean")

    # 把预测值 z 和真实值 y 传给损失函数，得到本次前向传播的损失值。
    # z、y 都有 2×3=6 个元素，所以当前损失为 6 个平方误差的平均值。
    # loss_value 是零维标量张量，形状为 torch.Size([])。
    loss_value = loss_fn(z, y)

    print("\n均方误差损失：")
    print(f"loss_value = {loss_value}")
    print(f"loss_value.shape = {loss_value.shape}")
    print(f"loss_value.grad_fn = {loss_value.grad_fn}")

    # ==================================================================
    # 第五部分：反向传播，计算 w 和 b 的梯度
    # ==================================================================

    # backward() 从 loss_value 出发，沿计算图反向应用链式法则：
    #
    #     loss_value → z = x @ w + b → w、b
    #
    # MSELoss(reduction="mean") 的 loss_value 已经是标量，因此可以直接写：
    #
    #     loss_value.backward()
    #
    # 这里保留 sum() 也不会改变结果，因为对单个标量求和仍是它本身。
    # 如果损失包含多个元素，sum() 会把它们聚合成一个标量；但 sum() 和
    # mean() 会导致不同的梯度尺度，真实任务中需要根据损失定义选择。
    loss_value.sum().backward()

    # backward() 完成后：
    #
    # w.grad.shape=(5, 3)，w 中每一个权重都有一个对应梯度。
    # b.grad.shape=(3,)，b 中每一个偏置都有一个对应梯度。
    #
    # 梯度表示参数发生微小变化时，loss 会向哪个方向、以多快速度变化。
    # 它不是更新后的参数，也不会自动替换 w 或 b。
    print("\n反向传播得到的 w.grad：")
    print(w.grad)
    print(f"w.grad.shape = {w.grad.shape}")

    print("\n反向传播得到的 b.grad：")
    print(b.grad)
    print(f"b.grad.shape = {b.grad.shape}")

    # 因为两条输入样本完全相同，而且 x 的所有元素都是 1，所以同一输出列中，
    # 5 个输入权重收到的梯度也相同。这可以在上面的 w.grad 中直接观察到。

    # ==================================================================
    # 第六部分：本示例没有执行参数更新
    # ==================================================================

    # 当前文件的目标是展示前向传播、损失计算和反向传播，因此到这里结束。
    # 如果需要执行一次手动梯度下降，可以使用下面的形式：
    #
    #     lr = 0.01
    #     with torch.no_grad():
    #         w -= lr * w.grad
    #         b -= lr * b.grad
    #
    # 多轮训练还需要在下一次 backward() 前清除旧梯度：
    #
    #     w.grad.zero_()
    #     b.grad.zero_()
    #
    # 正式项目通常使用 torch.optim，而不是手动更新参数。
