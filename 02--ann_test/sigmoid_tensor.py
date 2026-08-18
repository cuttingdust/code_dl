"""绘制Sigmoid原函数和导数，并演示PyTorch自动微分计算图。

这个示例主要学习四件事：

1. Sigmoid如何把任意实数压缩到 ``(0, 1)`` 范围。
2. Sigmoid在不同输入位置上的导数有什么变化。
3. PyTorch如何记录 ``x → sigmoid → y → sum → loss`` 计算图。
4. ``backward()`` 如何自动计算1000个输入点对应的Sigmoid导数。

Sigmoid公式：

    sigmoid(x) = 1 / (1 + exp(-x))

Sigmoid导数：

    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

它的主要特点：

* 原函数取值范围为 ``(0, 1)``，常用于二分类输出概率。
* 当 ``x=0`` 时，函数值为0.5、导数取得最大值0.25。
* 当输入绝对值很大时，函数进入饱和区，导数接近0。
* 深层网络反向传播时，许多小于1的导数连续相乘可能导致梯度消失。

本文件不是完整的模型训练示例：这里没有权重、标签和优化器；变量 ``loss``
只是为了把多元素输出聚合成标量，以便演示反向传播，并不是真实任务损失。
"""

import matplotlib.pyplot as plt
import torch


# 配置Matplotlib，使标题和坐标轴能够显示中文及负号。
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def print_computation_graph(grad_fn, indent: int = 0, visited=None) -> None:
    """从指定的 ``grad_fn`` 开始，递归打印自动微分计算图。

    Args:
        grad_fn:
            非叶子张量最后一步运算对应的反向传播节点。例如，
            ``torch.sigmoid(x)`` 的结果对应 ``SigmoidBackward0``。
        indent:
            当前节点的缩进空格数。递归越深，缩进越多，用来展示层级。
        visited:
            已经访问过的节点集合，用来防止共享节点被重复打印。

    叶子张量由用户直接创建，通常没有 ``grad_fn``，所以打印计算图时应从
    ``y.grad_fn`` 或 ``loss.grad_fn`` 等非叶子张量开始。
    """

    # 第一次调用没有传visited时，创建一个全新的空集合。
    # 不直接把set()写成默认参数，是为了避免多次调用共享可变默认对象。
    if visited is None:
        visited = set()

    # grad_fn=None表示已经没有更早的反向节点。
    # 节点已经访问过时也停止，防止重复遍历同一个计算分支。
    if grad_fn is None or grad_fn in visited:
        return

    visited.add(grad_fn)

    # 这里只打印节点的类型名称，例如SumBackward0、SigmoidBackward0。
    print(" " * indent + type(grad_fn).__name__)

    # next_functions保存当前反向节点所连接的上游反向节点。
    # 对每个上游节点递归调用，并增加缩进来显示计算图的层次。
    for next_fn, _ in grad_fn.next_functions:
        print_computation_graph(
            next_fn,
            indent + 4,
            visited,
        )


def sigmoid_demo() -> None:
    """绘制Sigmoid函数与导数，并打印对应的自动微分计算图。"""

    # subplots(1, 2)创建一行两列的绘图区域：
    #
    #     figure：整张画布
    #     graphs[0]：左侧子图，绘制Sigmoid原函数
    #     graphs[1]：右侧子图，绘制Sigmoid导数
    figure, graphs = plt.subplots(1, 2, figsize=(12, 5))

    # ==================================================================
    # 第一部分：绘制Sigmoid原函数
    # ==================================================================

    # linspace(-20, 20, steps=1000)在闭区间[-20, 20]中均匀取1000个点。
    # 整个x_function是长度为1000的一维张量，而每个x_function[i]是零维张量。
    # 1000个点之间有999个间隔，所以相邻点间距为40/999。
    #
    # 这里仅计算和绘图，不需要求导，所以不设置requires_grad=True。
    x_function = torch.linspace(-20, 20, steps=1000)

    # 对x_function中的1000个元素逐元素计算Sigmoid：
    #
    #     y_function[i] = 1 / (1 + exp(-x_function[i]))
    #
    # 输入和输出形状保持一致，都是torch.Size([1000])。
    y_function = torch.sigmoid(x_function)

    print("=" * 70)
    print("Sigmoid原函数数据")
    print(f"x_function.shape       = {x_function.shape}")
    print(f"y_function.shape       = {y_function.shape}")
    print(f"x范围                  = [{x_function[0].item()}, {x_function[-1].item()}]")
    print(f"y范围（采样结果）       = [{y_function.min().item():.10f}, "
          f"{y_function.max().item():.10f}]")
    print(f"x_function.requires_grad = {x_function.requires_grad}")
    print(f"y_function.grad_fn       = {y_function.grad_fn}")

    # x_function不需要梯度，所以PyTorch没有为这一部分建立自动微分计算图，
    # y_function.grad_fn因此是None。Matplotlib可以直接处理CPU张量。
    graphs[0].plot(x_function, y_function, color="blue", label="sigmoid(x)")
    graphs[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    graphs[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[0].set_xlabel("输入 x")
    graphs[0].set_ylabel("sigmoid(x)")
    graphs[0].set_title("Sigmoid原函数")
    graphs[0].grid(True)
    graphs[0].legend()

    # ==================================================================
    # 第二部分：建立需要自动求导的Sigmoid计算图
    # ==================================================================

    # 为了计算Sigmoid导数，重新创建一个requires_grad=True的输入张量。
    # x_gradient由用户直接创建，是叶子张量：
    #
    #     x_gradient.is_leaf = True
    #     x_gradient.grad_fn = None
    #
    # 叶子张量虽然没有grad_fn，但反向传播后的梯度会保存在它的.grad中。
    x_gradient = torch.linspace(
        -20,
        20,
        steps=1000,
        requires_grad=True,
    )

    # 前向传播：逐元素计算Sigmoid。
    # 因为输入x_gradient需要梯度，PyTorch会记录这次运算。
    # y_gradient是运算产生的非叶子张量，grad_fn为SigmoidBackward0。
    y_gradient = torch.sigmoid(x_gradient)

    print("\n" + "=" * 70)
    print("从y_gradient.grad_fn开始打印：x → sigmoid → y")
    print_computation_graph(y_gradient.grad_fn)

    # y_gradient含1000个元素，不能直接调用不带参数的y_gradient.backward()。
    # sum()把所有输出相加成一个零维标量，作为反向传播起点：
    #
    #     loss = y[0] + y[1] + ... + y[999]
    #
    # 这个sum不是模型训练中的真实损失函数，只是为了演示一次得到1000个
    # Sigmoid导数。由于每个y[i]只依赖对应x[i]，并且d(loss)/d(y[i])=1，
    # 所以最终x_gradient.grad[i]正好等于Sigmoid在x[i]处的导数。
    loss = y_gradient.sum()

    print("\n从loss.grad_fn开始打印完整计算图：x → sigmoid → y → sum → loss")
    print_computation_graph(loss.grad_fn)

    # 此时只是完成了前向计算和计算图建立，还没有真正计算梯度。
    print(f"\nbackward前 x_gradient.grad = {x_gradient.grad}")

    # 从标量loss出发反向传播：
    #
    #     SumBackward0
    #         ↓ 将上游梯度1传给每一个y元素
    #     SigmoidBackward0
    #         ↓ 计算sigmoid(x)*(1-sigmoid(x))
    #     AccumulateGrad
    #         ↓ 把结果累加到叶子张量x_gradient.grad
    loss.backward()

    print(f"backward后 x_gradient.grad.shape = {x_gradient.grad.shape}")
    print(f"最大导数                         = {x_gradient.grad.max().item():.10f}")
    print(f"x_gradient.is_leaf               = {x_gradient.is_leaf}")
    print(f"x_gradient.grad_fn               = {x_gradient.grad_fn}")
    print(f"y_gradient.is_leaf               = {y_gradient.is_leaf}")
    print(f"y_gradient.grad_fn               = {y_gradient.grad_fn}")
    print(f"loss.grad_fn                    = {loss.grad_fn}")

    # ==================================================================
    # 第三部分：用手算公式验证autograd的结果
    # ==================================================================

    # 手算公式：sigmoid'(x)=sigmoid(x)*(1-sigmoid(x))。
    # detach()表示验证计算不需要再建立一张新的自动微分计算图。
    manual_gradient = (
        y_gradient.detach() * (1 - y_gradient.detach())
    )

    # 比较自动微分结果与手算公式，最大误差应当接近0。
    max_error = (
        x_gradient.grad - manual_gradient
    ).abs().max().item()

    print(f"autograd与手算公式的最大误差      = {max_error:.12e}")

    # ==================================================================
    # 第四部分：绘制Sigmoid导数
    # ==================================================================

    # 绘图不需要参与自动求导，而且Matplotlib内部通常会转换为NumPy数组。
    # requires_grad=True的张量不能直接调用numpy()，所以需要：
    #
    #     detach()：脱离计算图
    #     cpu()：确保数据位于CPU，兼容将来使用GPU张量的情况
    #     numpy()：转换成Matplotlib能够直接使用的NumPy数组
    plot_x = x_gradient.detach().cpu().numpy()
    plot_gradient = x_gradient.grad.detach().cpu().numpy()

    graphs[1].plot(
        plot_x,
        plot_gradient,
        color="red",
        label="sigmoid'(x)",
    )
    graphs[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[1].axhline(0.25, color="gray", linestyle="--", linewidth=1)
    graphs[1].set_xlabel("输入 x")
    graphs[1].set_ylabel("sigmoid'(x)")
    graphs[1].set_title("Sigmoid导数（autograd计算）")
    graphs[1].grid(True)
    graphs[1].legend()

    # 设置整张画布的标题，并自动调整子图间距，避免标题和坐标文字重叠。
    figure.suptitle("Sigmoid函数及其导数")
    figure.tight_layout()
    plt.show()

def tanh_demo() -> None:
    """绘制Tanh原函数与导数，并验证PyTorch自动微分结果。

    Tanh（双曲正切）公式：

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    导数公式：

        tanh'(x) = 1 - tanh(x)²

    Tanh把任意实数压缩到(-1, 1)，并且以原点为中心：tanh(0)=0。
    当x=0时导数取得最大值1；当输入绝对值较大时，导数趋近0，
    因此Tanh在深层网络中也可能产生梯度消失。
    """

    # 创建一行两列的画布：左图绘制Tanh原函数，右图绘制Tanh导数。
    figure, graphs = plt.subplots(1, 2, figsize=(12, 5))

    # ==================================================================
    # 第一部分：绘制Tanh原函数
    # ==================================================================

    # 在闭区间[-20, 20]中均匀生成1001个点。
    # 使用奇数个点是为了让中间位置恰好包含x=0，便于观察：
    #
    #     tanh(0) = 0
    #     tanh'(0) = 1
    #
    # 这里只绘制函数，不需要自动求导，所以requires_grad保持默认False。
    x_function = torch.linspace(-20, 20, steps=1001)

    # torch.tanh()对每个元素分别计算双曲正切，输入输出形状相同。
    y_function = torch.tanh(x_function)

    center_index = len(x_function) // 2

    print("=" * 70)
    print("Tanh原函数数据")
    print(f"x_function.shape         = {x_function.shape}")
    print(f"y_function.shape         = {y_function.shape}")
    print(f"x范围                    = [{x_function[0].item()}, "
          f"{x_function[-1].item()}]")
    print(f"y范围（采样结果）         = [{y_function.min().item():.10f}, "
          f"{y_function.max().item():.10f}]")
    print(f"中间采样点x              = {x_function[center_index].item():.1f}")
    print(f"tanh(0)                  = {y_function[center_index].item():.1f}")
    print(f"x_function.requires_grad = {x_function.requires_grad}")
    print(f"y_function.grad_fn       = {y_function.grad_fn}")

    # 因为输入不要求梯度，这部分没有建立自动微分计算图，grad_fn为None。
    graphs[0].plot(x_function, y_function, color="blue", label="tanh(x)")
    graphs[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    graphs[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[0].axhline(1, color="lightgray", linestyle=":", linewidth=1)
    graphs[0].axhline(-1, color="lightgray", linestyle=":", linewidth=1)
    graphs[0].set_xlabel("输入 x")
    graphs[0].set_ylabel("tanh(x)")
    graphs[0].set_title("Tanh原函数")
    graphs[0].grid(True)
    graphs[0].legend()

    # ==================================================================
    # 第二部分：建立Tanh自动微分计算图
    # ==================================================================

    # 重新生成相同的1001个输入点，并设置requires_grad=True。
    # x_gradient是用户直接创建的叶子张量，反向传播后的梯度会保存在
    # x_gradient.grad中。
    x_gradient = torch.linspace(
        -20,
        20,
        steps=1001,
        requires_grad=True,
    )

    # 前向传播：PyTorch记录Tanh运算。
    # y_gradient是非叶子张量，其grad_fn为TanhBackward0。
    y_gradient = torch.tanh(x_gradient)

    print("\n" + "=" * 70)
    print("从y_gradient.grad_fn开始打印：x → tanh → y")
    print_computation_graph(y_gradient.grad_fn)

    # y_gradient有1001个元素。为了使用不带参数的backward()，先通过sum()
    # 聚合为零维标量。这里的loss只是反向传播起点，不是真实训练损失。
    # 因为d(sum(y))/d(y[i])=1，所以x_gradient.grad[i]就是每个位置的Tanh导数。
    loss = y_gradient.sum()

    print("\n从loss.grad_fn开始打印完整计算图：x → tanh → y → sum → loss")
    print_computation_graph(loss.grad_fn)

    print(f"\nbackward前 x_gradient.grad = {x_gradient.grad}")

    # 反向传播的节点顺序：
    #
    #     SumBackward0
    #         ↓ 把上游梯度1传给每一个y元素
    #     TanhBackward0
    #         ↓ 根据1-tanh(x)²计算每个位置的局部导数
    #     AccumulateGrad
    #         ↓ 把最终结果累加到叶子张量x_gradient.grad
    loss.backward()

    print(f"backward后 x_gradient.grad.shape = {x_gradient.grad.shape}")
    print(f"最大导数                          = {x_gradient.grad.max().item():.10f}")
    print(f"x=0处的导数                       = "
          f"{x_gradient.grad[center_index].item():.10f}")
    print(f"x_gradient.is_leaf               = {x_gradient.is_leaf}")
    print(f"x_gradient.grad_fn               = {x_gradient.grad_fn}")
    print(f"y_gradient.is_leaf               = {y_gradient.is_leaf}")
    print(f"y_gradient.grad_fn               = {y_gradient.grad_fn}")
    print(f"loss.grad_fn                     = {loss.grad_fn}")

    # ==================================================================
    # 第三部分：用Tanh导数公式验证autograd
    # ==================================================================

    # 手算公式：tanh'(x)=1-tanh(x)²。
    # detach()表示这次验证运算不需要加入原自动微分计算图。
    detached_y = y_gradient.detach()
    manual_gradient = 1 - detached_y**2

    # 自动微分结果与手算公式应该相同；只可能存在很小的浮点数舍入误差。
    max_error = (
        x_gradient.grad - manual_gradient
    ).abs().max().item()

    print(f"autograd与手算公式的最大误差      = {max_error:.12e}")

    # ==================================================================
    # 第四部分：绘制Tanh导数
    # ==================================================================

    # Matplotlib不参与自动求导，所以先detach()；cpu().numpy()同时兼容
    # 将来使用GPU张量的情况。
    plot_x = x_gradient.detach().cpu().numpy()
    plot_gradient = x_gradient.grad.detach().cpu().numpy()

    graphs[1].plot(
        plot_x,
        plot_gradient,
        color="red",
        label="tanh'(x)",
    )
    graphs[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    graphs[1].set_xlabel("输入 x")
    graphs[1].set_ylabel("tanh'(x)")
    graphs[1].set_title("Tanh导数（autograd计算）")
    graphs[1].grid(True)
    graphs[1].legend()

    figure.suptitle("Tanh函数及其导数")
    figure.tight_layout()
    plt.show()

def relu_demo() -> None:
    """绘制ReLU原函数与导数，并验证PyTorch自动微分结果。

    ReLU是Rectified Linear Unit（修正线性单元），公式为：

        relu(x) = max(0, x)

    它的分段形式为：

                    0，x <= 0
        relu(x) =
                    x，x > 0

    导数为：

                     0，x < 0
        relu'(x) =   未定义，x = 0
                     1，x > 0

    数学上ReLU在x=0处不可导，因为左导数是0、右导数是1。为了能够执行
    反向传播，深度学习框架需要选取一个约定值；PyTorch在x=0处使用梯度0。

    ReLU的主要优点：

    * 正半轴导数恒为1，缓解Sigmoid、Tanh在饱和区的梯度消失。
    * 计算只需要比较和取最大值，开销较小。
    * 负输入输出0，会产生稀疏激活。

    ReLU的主要问题是“神经元死亡”：如果某个神经元长期落在负半轴，梯度
    一直为0，参数可能无法继续更新。LeakyReLU等变体会给负半轴保留小斜率。
    """

    # 创建一行两列的画布：左图绘制ReLU原函数，右图绘制ReLU导数。
    figure, graphs = plt.subplots(1, 2, figsize=(12, 5))

    # ==================================================================
    # 第一部分：绘制ReLU原函数
    # ==================================================================

    # 在闭区间[-20, 20]中均匀生成1001个点。
    # 使用奇数个点确保中间索引对应x=0，方便观察ReLU在拐点处的输出和梯度。
    # 这里只绘图，不需要求导，因此requires_grad保持默认False。
    x_function = torch.linspace(-20, 20, steps=1001)

    # torch.relu()逐元素执行max(0, x)：
    #
    #     负数和0 → 输出0
    #     正数      → 保留原值
    #
    # 输入输出形状相同，都是torch.Size([1001])。
    y_function = torch.relu(x_function)

    center_index = len(x_function) // 2

    print("=" * 70)
    print("ReLU原函数数据")
    print(f"x_function.shape         = {x_function.shape}")
    print(f"y_function.shape         = {y_function.shape}")
    print(f"x范围                    = [{x_function[0].item()}, "
          f"{x_function[-1].item()}]")
    print(f"y范围（采样结果）         = [{y_function.min().item():.1f}, "
          f"{y_function.max().item():.1f}]")
    print(f"中间采样点x              = {x_function[center_index].item():.1f}")
    print(f"relu(0)                  = {y_function[center_index].item():.1f}")
    print(f"x_function.requires_grad = {x_function.requires_grad}")
    print(f"y_function.grad_fn       = {y_function.grad_fn}")

    # 输入不要求梯度，因此这部分不会建立自动微分计算图，grad_fn为None。
    graphs[0].plot(x_function, y_function, color="blue", label="relu(x)")
    graphs[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    graphs[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[0].set_xlabel("输入 x")
    graphs[0].set_ylabel("relu(x)")
    graphs[0].set_title("ReLU原函数")
    graphs[0].grid(True)
    graphs[0].legend()

    # ==================================================================
    # 第二部分：建立ReLU自动微分计算图
    # ==================================================================

    # 重新生成相同输入并设置requires_grad=True。
    # x_gradient是叶子张量，反向传播结果会保存在x_gradient.grad中。
    x_gradient = torch.linspace(
        -20,
        20,
        steps=1001,
        requires_grad=True,
    )

    # 前向传播：PyTorch记录ReLU运算。
    # y_gradient是非叶子张量，grad_fn通常为ReluBackward0。
    y_gradient = torch.relu(x_gradient)

    print("\n" + "=" * 70)
    print("从y_gradient.grad_fn开始打印：x → relu → y")
    print_computation_graph(y_gradient.grad_fn)

    # y_gradient包含1001个元素。sum()把它聚合成零维标量，使我们可以调用
    # 不带参数的backward()。这里的loss只是演示反向传播的标量起点，不是
    # 真实模型训练中的损失函数。
    loss = y_gradient.sum()

    print("\n从loss.grad_fn开始打印完整计算图：x → relu → y → sum → loss")
    print_computation_graph(loss.grad_fn)

    print(f"\nbackward前 x_gradient.grad = {x_gradient.grad}")

    # 反向传播的节点顺序：
    #
    #     SumBackward0
    #         ↓ 把上游梯度1传给每个y元素
    #     ReluBackward0
    #         ↓ x<0时乘0，x>0时乘1，x=0时PyTorch约定乘0
    #     AccumulateGrad
    #         ↓ 把最终结果累加到叶子张量x_gradient.grad
    loss.backward()

    print(f"backward后 x_gradient.grad.shape = {x_gradient.grad.shape}")
    print(f"负半轴梯度示例（x=-1附近）        = "
          f"{x_gradient.grad[center_index - 25].item():.1f}")
    print(f"x=0处PyTorch约定的梯度           = "
          f"{x_gradient.grad[center_index].item():.1f}")
    print(f"正半轴梯度示例（x=1附近）         = "
          f"{x_gradient.grad[center_index + 25].item():.1f}")
    print(f"x_gradient.is_leaf               = {x_gradient.is_leaf}")
    print(f"x_gradient.grad_fn               = {x_gradient.grad_fn}")
    print(f"y_gradient.is_leaf               = {y_gradient.is_leaf}")
    print(f"y_gradient.grad_fn               = {y_gradient.grad_fn}")
    print(f"loss.grad_fn                     = {loss.grad_fn}")

    # ==================================================================
    # 第三部分：用ReLU分段导数验证autograd
    # ==================================================================

    # 手动构造PyTorch采用的ReLU导数：
    #
    #     x > 0  → 1
    #     x <= 0 → 0（包括PyTorch在x=0处的约定）
    #
    # full_like创建与x_gradient形状、dtype和设备相同的全0或全1张量。
    detached_x = x_gradient.detach()
    manual_gradient = torch.where(
        detached_x > 0,
        torch.ones_like(detached_x),
        torch.zeros_like(detached_x),
    )

    # 自动微分结果应该和上述分段公式完全一致。
    max_error = (
        x_gradient.grad - manual_gradient
    ).abs().max().item()

    print(f"autograd与手算分段公式的最大误差  = {max_error:.12e}")

    # ==================================================================
    # 第四部分：绘制ReLU导数
    # ==================================================================

    # Matplotlib不参与自动求导，所以detach()后转为CPU NumPy数组。
    plot_x = x_gradient.detach().cpu().numpy()
    plot_gradient = x_gradient.grad.detach().cpu().numpy()

    graphs[1].plot(
        plot_x,
        plot_gradient,
        color="red",
        label="relu'(x)",
    )

    # 单独标出x=0处PyTorch采用的梯度0。
    graphs[1].scatter(
        [0],
        [0],
        color="black",
        zorder=3,
        label="PyTorch在x=0处取梯度0",
    )
    graphs[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    graphs[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    graphs[1].axhline(1, color="lightgray", linestyle=":", linewidth=1)
    graphs[1].set_ylim(-0.1, 1.1)
    graphs[1].set_xlabel("输入 x")
    graphs[1].set_ylabel("relu'(x)")
    graphs[1].set_title("ReLU导数（autograd计算）")
    graphs[1].grid(True)
    graphs[1].legend()

    figure.suptitle("ReLU函数及其导数")
    figure.tight_layout()
    plt.show()


def softmax_demo() -> None:
    """演示Softmax概率、维度选择、类别竞争和多分类反向传播。

    Softmax用于把一组任意实数分数（logits）转换成概率分布。对于第i类：

        softmax(z_i) = exp(z_i) / sum(exp(z_j))

    转换后的每个概率都在 ``(0, 1)`` 中，并且同一条样本所有类别概率之和为1。

    Softmax和Sigmoid、Tanh、ReLU有一个重要区别：

    * Sigmoid/Tanh/ReLU通常对每个元素独立计算。
    * Softmax需要同时查看同一组中的所有类别分数，一个类别概率变大时，
      其他类别概率通常会变小，因此类别之间存在竞争关系。

    多分类训练通常不要手动执行 ``softmax → log → NLL``，而应直接把原始
    logits传给 ``torch.nn.CrossEntropyLoss``。它内部完成数值更稳定的
    LogSoftmax和负对数似然计算。
    """

    # ==================================================================
    # 第一部分：准备两条样本、四个类别的原始分数logits
    # ==================================================================

    # logits是模型最后一个线性层直接输出的原始分数，还不是概率。
    #
    #     logits.shape = (2, 4)
    #
    # 2表示两条样本，4表示每条样本有四个候选类别。
    # 分数可以是任意实数，不要求在[0,1]内，也不要求总和为1。
    # requires_grad=True用于观察Softmax和交叉熵对logits的梯度。
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.1, -1.0],
            [0.5, 2.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    # 两条样本的真实类别下标：
    #
    #     第0条样本的真实类别是0
    #     第1条样本的真实类别是1
    #
    # CrossEntropyLoss要求标签使用整数类别下标，dtype必须为torch.long。
    targets = torch.tensor([0, 1], dtype=torch.long)

    print("=" * 70)
    print("Softmax输入数据")
    print(f"logits =\n{logits}")
    print(f"logits.shape         = {logits.shape}")
    print(f"logits.requires_grad = {logits.requires_grad}")
    print(f"targets              = {targets}")

    # ==================================================================
    # 第二部分：沿类别维度计算Softmax概率
    # ==================================================================

    # dim=1表示沿每一行的四个类别计算Softmax：每条样本分别得到一组概率。
    #
    #     输入：(2条样本, 4个类别)
    #     输出：(2条样本, 4个类别)
    #
    # 如果错误地使用dim=0，就会让不同样本在同一类别上共同归一化，得到的
    # 会是“每一列之和为1”，不符合通常的多分类概率含义。
    probabilities = torch.softmax(logits, dim=1)

    print("\n" + "=" * 70)
    print("Softmax概率")
    print(f"probabilities =\n{probabilities}")
    print(f"probabilities.shape = {probabilities.shape}")

    # 每一行代表一条样本的类别概率分布，所以每一行之和都应等于1。
    probability_sums = probabilities.sum(dim=1)
    print(f"每条样本的概率和    = {probability_sums}")

    # Softmax不会改变最大元素的位置：最大logit对应最大概率。
    predicted_from_logits = logits.argmax(dim=1)
    predicted_from_probabilities = probabilities.argmax(dim=1)
    print(f"根据logits预测类别  = {predicted_from_logits}")
    print(f"根据概率预测类别    = {predicted_from_probabilities}")

    # probabilities由需要梯度的logits计算而来，所以是非叶子张量，
    # grad_fn通常为SoftmaxBackward0。
    print(f"probabilities.is_leaf = {probabilities.is_leaf}")
    print(f"probabilities.grad_fn = {probabilities.grad_fn}")

    print("\n从probabilities.grad_fn开始打印：logits → softmax → probabilities")
    print_computation_graph(probabilities.grad_fn)

    # ==================================================================
    # 第三部分：验证Softmax的平移不变性
    # ==================================================================

    # 给同一条样本的所有类别分数同时加上同一个常数，不会改变Softmax概率：
    #
    #   exp(z_i+c) / sum(exp(z_j+c))
    # = exp(c)exp(z_i) / (exp(c)sum(exp(z_j)))
    # = exp(z_i) / sum(exp(z_j))
    #
    # PyTorch内部使用稳定算法，能够安全处理整体较大的logits。
    shifted_probabilities = torch.softmax(logits.detach() + 1000, dim=1)
    shift_error = (
        shifted_probabilities - probabilities.detach()
    ).abs().max().item()

    print("\nSoftmax平移不变性：")
    print(f"所有logits同时加1000后的最大概率误差 = {shift_error:.12e}")

    # ==================================================================
    # 第四部分：观察一个类别概率对所有logits的梯度
    # ==================================================================

    # 选择第0条样本的第0类概率p0作为标量输出。
    # 与逐元素激活不同，p0同时依赖这一行的全部四个logit。
    selected_probability = probabilities[0, 0]

    # torch.autograd.grad()返回指定输出对指定输入的梯度，但默认不会把结果
    # 累加到logits.grad。retain_graph=True保留这张Softmax计算图，便于后面
    # 继续使用probabilities做验证。
    selected_gradient = torch.autograd.grad(
        outputs=selected_probability,
        inputs=logits,
        retain_graph=True,
    )[0]

    print("\n" + "=" * 70)
    print("第0条样本的第0类概率对全部logits的梯度")
    print(f"selected_probability = {selected_probability.item():.10f}")
    print(f"selected_gradient =\n{selected_gradient}")

    # Softmax偏导公式：
    #
    #     当i=j：∂p_i/∂z_i = p_i(1-p_i)，对自己的logit是正梯度
    #     当i≠j：∂p_i/∂z_j = -p_i p_j，对其他类别logit是负梯度
    #
    # 这说明提高某一类别分数会提高自己的概率，同时压低其他类别的相对概率。
    sample_probabilities = probabilities[0].detach()
    manual_selected_gradient = -selected_probability.detach() * sample_probabilities
    manual_selected_gradient[0] = (
        selected_probability.detach()
        * (1 - selected_probability.detach())
    )

    selected_gradient_error = (
        selected_gradient[0] - manual_selected_gradient
    ).abs().max().item()

    print(f"Softmax偏导公式的最大误差 = {selected_gradient_error:.12e}")
    print(f"该行所有偏导之和         = {selected_gradient[0].sum().item():.12e}")

    # 不能用probabilities.sum().backward()来展示有意义的Softmax梯度。
    # 因为每条样本概率和恒等于1，是一个常数，对logits的梯度必然接近0。
    # 因此这里选择一个类别概率作为输出，才能观察类别之间的梯度关系。

    # ==================================================================
    # 第五部分：使用CrossEntropyLoss完成正式多分类损失和反向传播
    # ==================================================================

    # CrossEntropyLoss直接接收原始logits和整数类别标签。
    # 默认reduction="mean"，两条样本的损失会取平均：
    #
    #     loss = mean(-log(真实类别对应的Softmax概率))
    #
    # 不要先手动softmax再传入CrossEntropyLoss，否则会重复归一化。
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    classification_loss = loss_fn(logits, targets)

    print("\n" + "=" * 70)
    print("CrossEntropyLoss多分类训练")
    print(f"classification_loss = {classification_loss.item():.10f}")
    print(f"loss.shape           = {classification_loss.shape}")
    print(f"loss.grad_fn         = {classification_loss.grad_fn}")
    print("CrossEntropyLoss计算图：")
    print_computation_graph(classification_loss.grad_fn)

    # 反向传播前，logits是叶子张量但还没有梯度。
    print(f"backward前 logits.grad = {logits.grad}")
    classification_loss.backward()
    print(f"backward后 logits.grad =\n{logits.grad}")

    # CrossEntropyLoss对logits的梯度公式为：
    #
    #     (softmax概率 - 真实类别one-hot向量) / batch_size
    #
    # 因为当前loss对两条样本取平均，所以最后除以2。
    manual_loss_gradient = probabilities.detach().clone()
    row_indices = torch.arange(len(targets))
    manual_loss_gradient[row_indices, targets] -= 1
    manual_loss_gradient /= len(targets)

    loss_gradient_error = (
        logits.grad - manual_loss_gradient
    ).abs().max().item()

    print(f"交叉熵梯度公式的最大误差 = {loss_gradient_error:.12e}")

    # ==================================================================
    # 第六部分：绘制两条样本的类别概率和Softmax局部梯度
    # ==================================================================

    figure, graphs = plt.subplots(1, 2, figsize=(12, 5))
    class_indices = torch.arange(logits.shape[1])
    bar_width = 0.35

    # 左图：对比两条样本在四个类别上的概率分布。
    probability_values = probabilities.detach().cpu().numpy()
    class_values = class_indices.cpu().numpy()

    graphs[0].bar(
        class_values - bar_width / 2,
        probability_values[0],
        width=bar_width,
        label="样本0",
    )
    graphs[0].bar(
        class_values + bar_width / 2,
        probability_values[1],
        width=bar_width,
        label="样本1",
    )
    graphs[0].set_xticks(class_values)
    graphs[0].set_xlabel("类别下标")
    graphs[0].set_ylabel("Softmax概率")
    graphs[0].set_title("每条样本的类别概率分布")
    graphs[0].set_ylim(0, 1)
    graphs[0].grid(True, axis="y")
    graphs[0].legend()

    # 右图：第0类概率对四个类别logit的梯度。
    # 自己类别的梯度为正，其他类别的梯度为负，直观体现类别竞争关系。
    local_gradient_values = selected_gradient[0].detach().cpu().numpy()
    bar_colors = ["green" if value >= 0 else "red" for value in local_gradient_values]

    graphs[1].bar(
        class_values,
        local_gradient_values,
        color=bar_colors,
    )
    graphs[1].axhline(0, color="black", linewidth=1)
    graphs[1].set_xticks(class_values)
    graphs[1].set_xlabel("logit类别下标")
    graphs[1].set_ylabel("偏导数")
    graphs[1].set_title("样本0的p(class=0)对各logit的梯度")
    graphs[1].grid(True, axis="y")

    figure.suptitle("Softmax概率与类别竞争关系")
    figure.tight_layout()
    plt.show()



if __name__ == "__main__":
    # sigmoid激活函数
    # sigmoid_demo()

    # tanh激活函数
    # tanh_demo()

    # relu激活函数
    # relu_demo()

    # softmax激活函数
    softmax_demo()
