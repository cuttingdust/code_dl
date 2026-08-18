import torch
from torch import optim

"""
梯度下降优化算法示例
====================

一、优化器主要是用来做什么的？
    神经网络训练通常会不断重复下面三个步骤：

        1. loss.backward()：计算损失函数对每个参数的梯度；
        2. optimizer.step()：根据梯度和优化算法更新参数；
        3. optimizer.zero_grad()：清除旧梯度，为下一轮计算做准备。

    注意：
        反向传播负责“计算梯度”，优化器负责“使用梯度更新参数”。
        优化器不会改变损失函数，也不会代替反向传播。

二、本示例为什么使用 loss = 2 * w ** 2？
    对这个损失函数求导：

        loss = 2w²
        d(loss)/dw = 4w

    它的最小值位于 w=0。因此，可以观察不同优化器怎样把初始值 w=1
    逐渐移动到 0 附近。

三、四种算法最核心的区别
    1. Momentum：
       保存“过去梯度形成的速度”，沿着持续一致的方向加速。

    2. Adagrad：
       累加“从训练开始到现在的梯度平方”，为每个参数分别调整学习率。
       某个参数过去更新得越多，它后面的有效学习率就越小。

    3. RMSprop：
       保存“近期梯度平方的移动平均”，不会永久保留非常久以前的梯度。
       它解决了 Adagrad 有效学习率可能过早衰减的问题。

    4. Adam：
       同时保存“梯度的移动平均”和“梯度平方的移动平均”。
       可以粗略理解为 Momentum 与 RMSprop 思想的结合。

四、简单的选择建议
    - SGD + Momentum：
      结构简单、内存占用较少，在很多视觉任务中经过仔细调参后表现很好。

    - Adagrad：
      参数或特征非常稀疏时比较有价值，例如部分文本和推荐系统任务；
      但长时间训练时，有效学习率可能变得过小。

    - RMSprop：
      梯度变化较大或目标不断变化时可以尝试，历史上也常用于循环神经网络。

    - Adam：
      通常是新任务很方便的起点，对学习率没有调好的情况相对宽容；
      但它并不保证在所有任务上都优于 SGD + Momentum。

    实际项目不存在永远最好的优化器，最终仍需要通过验证集结果进行选择。
"""


def momentum_demo() -> None:
    """
    Momentum（动量法）
    ----------------
    普通 SGD 只看当前梯度：

        w = w - lr * grad

    PyTorch Momentum 还会维护一个 momentum_buffer（动量缓冲区）：

        velocity = momentum * previous_velocity + grad
        w = w - lr * velocity

    这里没有写公式中常见的 (1-momentum) * grad，是因为 PyTorch 的
    SGD Momentum 默认采用上面的实现形式。

    作用：
        - 连续多轮梯度方向相同时，动量会逐渐积累，从而加快更新；
        - 梯度频繁左右摆动时，正负方向会互相抵消，从而减小振荡；
        - 即使当前梯度恰好为 0，过去积累的动量仍可能让参数继续移动。

    可能的问题：
        动量太大或学习率太高时，参数可能冲过最低点，在最低点两侧振荡。
    """

    # 1- 初始化一个需要训练的标量参数 w。
    # requires_grad=True 表示 PyTorch 需要记录与 w 有关的计算，以便求梯度。
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    # 2- 定义梯度下降算法：Momentum动量法
    # params=[w]：告诉优化器需要更新参数 w。
    # lr=0.1：基础学习率，控制每次更新的总体步长。
    # momentum=0.9：保留上一轮动量的 90%。
    # 如果 momentum=0，那么 optim.SGD 就退化成普通 SGD。
    optimizer = optim.SGD(params=[w], lr=0.1, momentum=0.9)

    # 3- 循环更新 w。
    for i in range(5):
        # 保存更新前的参数值，便于和更新后的值进行比较。
        old_w = w.item()

        # 3.1- 定义损失函数。
        # loss=2w²，梯度为 4w，理论最低点为 w=0。
        loss = 2 * w ** 2

        # 3.2- 反向传播的标准流程
        # 清除上一轮保存在 w.grad 中的梯度。
        # 这里只会清除梯度，不会清除优化器内部的 momentum_buffer。
        optimizer.zero_grad()

        # loss 是只有一个元素的 0 维标量张量，可以直接调用 backward()。
        # 执行后，当前梯度 4w 会保存到 w.grad 中。
        loss.backward()

        # 先保存当前梯度。optimizer.step() 更新参数，但不会把 w.grad 清空。
        grad = w.grad.item()

        # 使用“当前梯度 + 历史动量”更新 w。
        # 所以参数实际变化量不一定等于 lr * 当前打印出来的 grad。
        optimizer.step()

        print(
            f"第{i + 1}次：更新前w={old_w:.6f}，"
            f"grad={grad:.6f}，更新后w={w.item():.6f}"
        )


def adagrad_demo() -> None:
    """
    Adagrad（自适应梯度）
    -------------------
    Adagrad 会为每一个参数累计历史梯度的平方：

        sum_square = sum_square + grad²
        w = w - lr * grad / (sqrt(sum_square) + eps)

    eps 是一个很小的数，用来避免分母为 0。

    作用：
        每个参数拥有不同的“有效学习率”。很少出现梯度的稀疏参数，
        累计的梯度平方较小，因此仍可以获得相对较大的更新。

    可能的问题：
        sum_square 只增不减，所以有效学习率会越来越小。
        训练时间很长时，参数可能几乎停止更新。

    与 Momentum 的区别：
        Momentum 累积的是梯度方向，用来形成速度；
        Adagrad 累积的是梯度平方，用来缩放每个参数的学习率。
    """

    # 1- 初始化需要优化的参数 w。
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    # 2- Adagrad优化器
    # lr=0.1 是初始基础学习率，实际更新步长还会除以历史梯度平方和的平方根。
    optimizer = optim.Adagrad(params=[w], lr=0.1)

    # 3- 循环更新 w。
    for i in range(5):
        old_w = w.item()
        loss = 2 * w ** 2

        # 清除 w.grad，但不会清除 Adagrad 内部累计的 sum（梯度平方和）。
        optimizer.zero_grad()
        loss.backward()
        grad = w.grad.item()

        # 根据当前梯度以及历史梯度平方和更新参数。
        optimizer.step()

        print(
            f"第{i + 1}次：更新前w={old_w:.6f}，"
            f"grad={grad:.6f}，更新后w={w.item():.6f}"
        )


def rmsprop_demo() -> None:
    """
    RMSprop（梯度平方的指数移动平均）
    --------------------------------
    RMSprop 不像 Adagrad 那样永久累加所有梯度平方，而是更加关注近期数据：

        avg_square = alpha * previous_avg_square + (1-alpha) * grad²
        w = w - lr * grad / (sqrt(avg_square) + eps)

    作用：
        - 为不同参数自适应地缩放更新步长；
        - 逐渐遗忘很久以前的梯度，避免有效学习率一直缩小到接近 0；
        - 梯度较大时减小更新，梯度较小时相对放大更新。

    与 Adagrad 的区别：
        Adagrad 使用“所有历史梯度平方的总和”；
        RMSprop 使用“近期梯度平方的移动平均”，旧信息会逐渐衰减。

    这里没有启用 RMSprop 自己的 momentum 参数，以便单独观察 RMSprop
    对梯度平方进行自适应缩放的作用。
    """

    # 1- 初始化需要优化的参数 w。
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    # 2- 定义 RMSprop 优化器。
    # alpha=0.9：保留上一轮梯度平方平均值的 90%，关注近期梯度变化。
    # 注意：部分教材把这个系数写成 beta，PyTorch 的参数名是 alpha。
    optimizer = optim.RMSprop(params=[w], lr=0.1, alpha=0.9)

    # 3- 循环更新 w。
    for i in range(10):
        old_w = w.item()
        loss = 2 * w ** 2

        # zero_grad() 只清除 w.grad，不清除 RMSprop 内部的 avg_square。
        optimizer.zero_grad()
        loss.backward()
        grad = w.grad.item()

        # 使用当前梯度和近期梯度平方的移动平均更新参数。
        optimizer.step()

        print(
            f"第{i + 1}次：更新前w={old_w:.6f}，"
            f"grad={grad:.6f}，更新后w={w.item():.6f}"
        )


def adam_demo() -> None:
    """
    Adam（Adaptive Moment Estimation）
    ---------------------------------
    Adam 同时维护两类历史信息：

        1. 一阶矩 m：梯度的指数移动平均，类似 Momentum 的“方向和速度”；
        2. 二阶矩 v：梯度平方的指数移动平均，类似 RMSprop 的“步长缩放”。

    简化后的思想：

        m = beta1 * previous_m + (1-beta1) * grad
        v = beta2 * previous_v + (1-beta2) * grad²
        w = w - lr * corrected_m / (sqrt(corrected_v) + eps)

    实际 Adam 还会对训练初期偏小的 m、v 做“偏差修正”。

    作用：
        - 利用一阶矩平滑梯度方向；
        - 利用二阶矩为每个参数自适应调整更新步长；
        - 通常不需要像普通 SGD 那样进行大量学习率调试，就能较快开始训练。

    可能的问题：
        - 每个参数要额外保存 m 和 v，比 SGD 占用更多优化器内存；
        - 它不是任何任务上都一定最好，部分任务中 SGD + Momentum 最终泛化更好。

    与前三种算法的关系：
        Momentum：主要利用一阶矩；
        RMSprop：主要利用二阶矩；
        Adam：同时利用一阶矩和二阶矩。
    """

    # 1- 初始化需要优化的参数 w。
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    # 2- 定义 Adam 优化器。
    # betas=(0.9, 0.999)：
    #   beta1=0.9   -> 一阶矩（梯度移动平均）的衰减系数；
    #   beta2=0.999 -> 二阶矩（梯度平方移动平均）的衰减系数。
    optimizer = optim.Adam(params=[w], lr=0.1, betas=(0.9, 0.999))

    # 3- 循环更新 w。
    for i in range(10):
        old_w = w.item()
        loss = 2 * w ** 2

        # 只清除 w.grad，不会清除 Adam 内部保存的一阶矩 m 和二阶矩 v。
        optimizer.zero_grad()
        loss.backward()
        grad = w.grad.item()

        # 使用当前梯度、历史一阶矩和历史二阶矩共同更新参数。
        optimizer.step()

        print(
            f"第{i + 1}次：更新前w={old_w:.6f}，"
            f"grad={grad:.6f}，更新后w={w.item():.6f}"
        )


if __name__ == '__main__':
    # 根据需要取消对应函数前面的注释，可以分别观察四种算法。

    # 1- Momentum
    print("Momentum", "-" * 30)
    momentum_demo()

    # 2- Adagrad
    print("Adagrad", "-" * 30)
    adagrad_demo()

    # 3- RMSprop
    print("RMSprop", "-" * 30)
    rmsprop_demo()

    # 4- Adam
    print("Adam", "-" * 30)
    adam_demo()
