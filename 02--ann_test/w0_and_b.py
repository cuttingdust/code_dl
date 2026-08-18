"""
    参数初始化总结
        1- 从激活函数搭配的角度
            Sigmoid/Tanh：推荐使用xavier初始化方式
            relu系列：推荐使用kaiming初始化方式
        2- 从神经网络的层数的角度
            浅层网络（隐藏层层数<=10）：多种初始化方式都可以使用。推荐使用kaiming
            深层网络（隐藏层层数>10）：使用kaiming或者xavier初始化方式都行
"""
import torch
from torch import nn
from torch.ao.nn.quantized.functional import linear


def demo01() -> None:
    """演示使用均匀分布初始化线性层的权重和偏置。

    均匀分布 ``U(a, b)`` 的含义是：在区间 ``[a, b)`` 中随机取数，
    区间内相同长度的小区间被抽中的概率相同。

    例如使用 ``U(0, 1)`` 时：

        落入[0.0, 0.2)的概率约为20%
        落入[0.2, 0.4)的概率约为20%
        落入[0.8, 1.0)的概率约为20%

    “均匀”说的是概率密度均匀，并不是说生成的有限个数字之间距离相等。
    随机结果中可能有两个值很接近，也可能某一小段暂时没有值。

    均匀分布的理论统计量：

        均值     E[X]   = (a + b) / 2
        方差     Var[X] = (b - a)² / 12
        标准差   Std[X] = (b - a) / sqrt(12)

    注意：这个函数主要用于理解 ``uniform_``，并不是在说明所有神经网络
    都应该把权重和偏置初始化为U(0,1)。正式网络通常根据激活函数和输入
    维度选择Xavier、Kaiming等初始化；偏置也经常初始化为0。
    """

    # 固定PyTorch随机种子，使每次运行这个教学示例都得到相同的随机值。
    # 随机种子只用于复现实验，并不会让生成的数据失去随机分布特征。
    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建一个线性层
    # ==================================================================

    # nn.Linear实现：
    #
    #     output = input @ weight.T + bias
    #
    # in_features=5：每条输入样本有5个特征。
    # out_features=3：每条样本产生3个输出。
    # bias=True：使用可训练偏置，这是nn.Linear的默认设置。
    #
    # 因此参数形状为：
    #
    #     linear.weight.shape = (3, 5)，共有3×5=15个权重
    #     linear.bias.shape   = (3,)，共有3个偏置
    linear = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    # 创建nn.Linear时，PyTorch已经自动执行了一次默认初始化。
    # 下面调用nn.init.uniform_会原地覆盖这些默认参数，而不是在原值上相加。
    print("=" * 70)
    print("nn.Linear创建后的默认参数（即将被uniform_覆盖）")
    print(f"默认weight =\n{linear.weight}")
    print(f"默认bias   =\n{linear.bias}")

    # ==================================================================
    # 第2步：指定均匀分布上下界
    # ==================================================================

    # nn.init.uniform_的函数形式为：
    #
    #     nn.init.uniform_(tensor, a=0.0, b=1.0)
    #
    # a是下界，b是上界；不写a和b时默认使用U(0,1)。
    # 这里显式写出来，是为了清楚展示当前随机范围。
    lower_bound = 0.0
    upper_bound = 1.0

    # ==================================================================
    # 第3步：使用U(0,1)覆盖权重和偏置
    # ==================================================================

    # uniform_名称末尾的下划线“_”是PyTorch的惯例，表示原地操作：
    #
    #     它直接修改linear.weight保存的数据；
    #     不会创建一个新的Parameter并替换变量；
    #     返回值仍然引用传入的同一个张量。
    #
    # nn.init中的初始化函数会在不记录梯度的模式下修改参数，不会把初始化
    # 操作加入autograd计算图。初始化后Parameter仍然requires_grad=True。
    returned_weight = nn.init.uniform_(
        linear.weight,
        a=lower_bound,
        b=upper_bound,
    )

    returned_bias = nn.init.uniform_(
        linear.bias,
        a=lower_bound,
        b=upper_bound,
    )

    print("\n" + "=" * 70)
    print(f"使用均匀分布U({lower_bound}, {upper_bound})初始化后的参数")
    print(f"weight =\n{linear.weight}")
    print(f"bias   =\n{linear.bias}")

    # 验证uniform_确实是原地操作：返回张量与原Parameter具有相同数据地址。
    print("\nuniform_原地操作检查：")
    print(
        "returned_weight与linear.weight共享数据地址 = "
        f"{returned_weight.data_ptr() == linear.weight.data_ptr()}"
    )
    print(
        "returned_bias与linear.bias共享数据地址     = "
        f"{returned_bias.data_ptr() == linear.bias.data_ptr()}"
    )

    # 初始化不会关闭参数的梯度功能。模型训练时，反向传播仍会把梯度保存到
    # linear.weight.grad和linear.bias.grad中。
    print(f"weight.requires_grad = {linear.weight.requires_grad}")
    print(f"bias.requires_grad   = {linear.bias.requires_grad}")

    # ==================================================================
    # 第4步：检查实际取值范围和样本统计量
    # ==================================================================

    # detach()表示下面只观察参数数值，不需要建立新的自动微分计算图。
    weight_values = linear.weight.detach()
    bias_values = linear.bias.detach()

    print("\n实际取值范围：")
    print(
        f"weight最小值={weight_values.min().item():.6f}，"
        f"最大值={weight_values.max().item():.6f}"
    )
    print(
        f"bias最小值={bias_values.min().item():.6f}，"
        f"最大值={bias_values.max().item():.6f}"
    )

    # 所有值都应该满足a <= value < b。
    weight_in_range = bool(
        ((weight_values >= lower_bound) & (weight_values < upper_bound))
        .all()
        .item()
    )
    bias_in_range = bool(
        ((bias_values >= lower_bound) & (bias_values < upper_bound))
        .all()
        .item()
    )

    print(f"所有weight都在[a,b)内 = {weight_in_range}")
    print(f"所有bias都在[a,b)内   = {bias_in_range}")

    theoretical_mean = (lower_bound + upper_bound) / 2
    theoretical_std = (upper_bound - lower_bound) / (12**0.5)

    print("\n理论值与本次小样本统计：")
    print(f"理论均值             = {theoretical_mean:.6f}")
    print(f"15个weight的实际均值 = {weight_values.mean().item():.6f}")
    print(f"理论标准差           = {theoretical_std:.6f}")
    print(
        "15个weight的实际标准差 = "
        f"{weight_values.std(unbiased=False).item():.6f}"
    )

    # 这里只抽取15个权重，实际均值和标准差不一定非常接近理论值。
    # 随机样本数量越多，整体直方图和统计量通常越接近理论均匀分布。

    # ==================================================================
    # 第5步：如何修改成对称均匀分布
    # ==================================================================

    # 当前U(0,1)只会生成非负数，主要用于讲解uniform_接口。
    # 如果希望围绕0对称，可以改成：
    #
    #     nn.init.uniform_(linear.weight, a=-0.5, b=0.5)
    #
    # 这样理论均值为0。但真实网络中上下界通常不应随意指定，而应结合
    # fan_in、fan_out和激活函数选择Xavier或Kaiming初始化。

def demo02() -> None:
    """演示使用正态分布初始化线性层的权重和偏置。

    正态分布通常写作 ``N(mean, std²)``：

        mean：均值，决定钟形曲线的中心位置
        std：标准差，决定随机数围绕均值的分散程度
        std²：方差

    ``nn.init.normal_`` 的参数形式为：

        nn.init.normal_(tensor, mean=0.0, std=1.0)

    默认得到标准正态分布 ``N(0,1)``。它的数值不是均匀散布，而是越靠近
    均值0出现概率越高、离0越远出现概率越低，整体图像类似钟形。

    标准正态分布常见的“68-95-99.7规律”：

        约68%的值位于 mean ± 1×std
        约95%的值位于 mean ± 2×std
        约99.7%的值位于 mean ± 3×std

    正态分布理论取值范围是整个实数轴 ``(-∞,+∞)``，所以它不像均匀分布
    那样有严格上下界；只是距离均值非常远的值出现概率很低。

    注意：这个函数主要讲解normal_接口。直接用N(0,1)初始化真实深层网络
    可能使参数和激活值过大；实际通常使用Xavier normal、Kaiming normal等
    根据fan_in/fan_out自动缩放标准差的初始化方法，偏置则经常初始化为0。
    """

    # 固定随机种子，使每次独立运行demo02时得到相同结果，方便对比和调试。
    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建线性层并查看PyTorch默认初始化
    # ==================================================================

    # 线性层执行：output = input @ weight.T + bias。
    # 5个输入特征、3个输出，因此：
    #
    #     weight.shape = (3, 5)，共15个权重
    #     bias.shape   = (3,)，共3个偏置
    linear = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    # nn.Linear构造时已经调用reset_parameters()完成默认随机初始化。
    # 下面只是在手动normal_覆盖前观察原始参数。
    print("=" * 70)
    print("nn.Linear创建后的默认参数（即将被normal_覆盖）")
    print(f"默认weight =\n{linear.weight}")
    print(f"默认bias   =\n{linear.bias}")

    # ==================================================================
    # 第2步：指定正态分布的均值和标准差
    # ==================================================================

    # 不传参数时，normal_默认mean=0、std=1。
    # 这里显式写出来，让分布含义更加清楚。
    mean = 0.0
    std = 1.0

    # 方差等于标准差的平方。normal_接收的是std，而不是variance（方差）。
    variance = std**2

    print("\n正态分布参数：")
    print(f"mean（均值）     = {mean}")
    print(f"std（标准差）    = {std}")
    print(f"variance（方差） = {variance}")

    # ==================================================================
    # 第3步：使用N(0,1)原地覆盖权重和偏置
    # ==================================================================

    # normal_末尾的下划线表示原地操作：它直接修改Parameter内部的数据，
    # 不会创建新Parameter，也不会改变linear.weight/bias的对象身份。
    # nn.init初始化函数在不记录梯度的模式下操作，不会把初始化加入计算图。
    returned_weight = nn.init.normal_(
        linear.weight,
        mean=mean,
        std=std,
    )

    returned_bias = nn.init.normal_(
        linear.bias,
        mean=mean,
        std=std,
    )

    print("\n" + "=" * 70)
    print(f"使用正态分布N({mean}, {variance})初始化后的参数")
    print(f"weight =\n{linear.weight}")
    print(f"bias   =\n{linear.bias}")

    # 返回值和原Parameter引用相同底层数据，证明normal_是原地操作。
    print("\nnormal_原地操作检查：")
    print(
        "returned_weight与linear.weight共享数据地址 = "
        f"{returned_weight.data_ptr() == linear.weight.data_ptr()}"
    )
    print(
        "returned_bias与linear.bias共享数据地址     = "
        f"{returned_bias.data_ptr() == linear.bias.data_ptr()}"
    )

    # 初始化只改变参数数值，不会关闭自动求导。训练时仍会产生对应.grad。
    print(f"weight.requires_grad = {linear.weight.requires_grad}")
    print(f"bias.requires_grad   = {linear.bias.requires_grad}")

    # ==================================================================
    # 第4步：查看小样本的实际统计量
    # ==================================================================

    # detach()表示下面只读取数据，不需要建立额外自动微分计算图。
    weight_values = linear.weight.detach()
    bias_values = linear.bias.detach()

    print("\n15个weight的实际统计：")
    print(f"最小值   = {weight_values.min().item():.6f}")
    print(f"最大值   = {weight_values.max().item():.6f}")
    print(f"实际均值 = {weight_values.mean().item():.6f}")
    print(
        "实际标准差 = "
        f"{weight_values.std(unbiased=False).item():.6f}"
    )

    print("\n3个bias的实际统计：")
    print(f"最小值   = {bias_values.min().item():.6f}")
    print(f"最大值   = {bias_values.max().item():.6f}")
    print(f"实际均值 = {bias_values.mean().item():.6f}")

    # 只有15个权重和3个偏置，样本太少，实际均值/标准差可能与理论值差异较大。
    # 这并不表示normal_错误；分布规律需要大量随机样本才能明显体现。

    # ==================================================================
    # 第5步：用大量独立样本验证正态分布规律
    # ==================================================================

    # 额外生成100000个数只用于统计演示，不属于linear模型参数。
    # torch.empty()先分配未初始化内存，再由normal_原地填充正态随机数。
    sample_count = 100_000
    large_sample = torch.empty(sample_count)
    nn.init.normal_(large_sample, mean=mean, std=std)

    sample_mean = large_sample.mean().item()
    sample_std = large_sample.std(unbiased=False).item()

    # 计算落在均值左右1、2、3个标准差范围内的样本比例。
    within_one_std = (
        (large_sample >= mean - std)
        & (large_sample <= mean + std)
    ).float().mean().item()

    within_two_std = (
        (large_sample >= mean - 2 * std)
        & (large_sample <= mean + 2 * std)
    ).float().mean().item()

    within_three_std = (
        (large_sample >= mean - 3 * std)
        & (large_sample <= mean + 3 * std)
    ).float().mean().item()

    print(f"\n{sample_count}个独立样本的统计：")
    print(f"理论均值={mean:.6f}，实际均值={sample_mean:.6f}")
    print(f"理论标准差={std:.6f}，实际标准差={sample_std:.6f}")
    print(f"落在mean±1std内：{within_one_std:.2%}（理论约68.27%）")
    print(f"落在mean±2std内：{within_two_std:.2%}（理论约95.45%）")
    print(f"落在mean±3std内：{within_three_std:.2%}（理论约99.73%）")

    # ==================================================================
    # 第6步：如何调整正态分布
    # ==================================================================

    # 如果写成：
    #
    #     nn.init.normal_(linear.weight, mean=0.0, std=0.01)
    #
    # 均值仍为0，但随机数会更集中在0附近。std越小，分布越窄；std越大，
    # 分布越宽、出现绝对值较大参数的概率越高。
    #
    # 正式网络不应只凭感觉选择std，通常让Xavier/Kaiming根据网络结构计算。

def demo03() -> None:
    """演示全零初始化，以及为什么通常只把偏置初始化为0。

    ``nn.init.zeros_(tensor)`` 会把传入张量的所有元素原地设置为0。
    它不是随机分布，没有均值、标准差等需要选择的分布参数。

    常见实践：

    * 偏置bias通常可以初始化为0。不同神经元仍有不同的随机权重，因此它们
      能产生不同输出、接收不同梯度，偏置也会在训练中逐渐离开0。
    * 普通隐藏层的weight通常不能全部初始化为0。如果同层神经元的参数和
      计算完全相同，它们可能得到相同输出与相同梯度，之后仍保持相同，
      相当于多个神经元只发挥一个神经元的作用，这叫作“对称性问题”。

    “不能零初始化权重”不是对所有数学模型都绝对禁止；某些特殊结构会有意
    零初始化特定层。但对于初学者编写的普通全连接/卷积隐藏层，应使用随机
    Xavier、Kaiming等方法初始化weight，bias则常用zeros_初始化。
    """

    # 固定随机种子，使线性层的默认随机权重能够复现。
    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建线性层并观察默认参数
    # ==================================================================

    # nn.Linear计算：output = input @ weight.T + bias。
    #
    #     5个输入特征 → weight.shape=(3,5)
    #     3个输出     → bias.shape=(3,)
    #
    # 构造函数已经为weight和bias执行了默认随机初始化。
    linear = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    print("=" * 70)
    print("nn.Linear创建后的默认参数")
    print(f"默认weight =\n{linear.weight}")
    print(f"默认bias   =\n{linear.bias}")

    # 保存weight的独立副本，用于证明稍后只清零bias，不会改变weight。
    # detach()脱离计算图，clone()复制独立数据。
    weight_before = linear.weight.detach().clone()

    # 同时记录bias对象身份和底层数据地址，验证zeros_是原地操作。
    bias_id_before = id(linear.bias)
    bias_data_ptr_before = linear.bias.data_ptr()

    # ==================================================================
    # 第2步：只把偏置bias初始化为0
    # ==================================================================

    # zeros_末尾的下划线表示原地操作：
    #
    #     直接把linear.bias已有内存中的每个元素写成0；
    #     不会创建新的Parameter；
    #     不会把初始化操作加入autograd计算图；
    #     不会关闭bias.requires_grad。
    returned_bias = nn.init.zeros_(linear.bias)

    print("\n" + "=" * 70)
    print("执行nn.init.zeros_(linear.bias)之后")
    print(f"weight =\n{linear.weight}")
    print(f"bias   =\n{linear.bias}")

    # torch.equal要求形状和值完全相同。结果True说明weight没有被修改。
    print("\n零偏置初始化检查：")
    print(
        "weight是否保持不变                 = "
        f"{torch.equal(weight_before, linear.weight.detach())}"
    )
    print(
        "bias是否全部为0                    = "
        f"{bool((linear.bias == 0).all().item())}"
    )
    print(
        "bias的Python对象id是否不变          = "
        f"{bias_id_before == id(linear.bias)}"
    )
    print(
        "bias的底层数据地址是否不变          = "
        f"{bias_data_ptr_before == linear.bias.data_ptr()}"
    )
    print(
        "zeros_返回值是否共享同一数据地址    = "
        f"{returned_bias.data_ptr() == linear.bias.data_ptr()}"
    )
    print(f"bias.requires_grad               = {linear.bias.requires_grad}")

    # ==================================================================
    # 第3步：证明偏置从0开始仍然可以学习
    # ==================================================================

    # 准备2条全1样本，每条样本有5个特征。
    sample_input = torch.ones(2, 5)

    # 因为bias当前为0，前向传播等价于input @ weight.T。
    # 随机weight使三个输出神经元仍然得到不同结果，不会因为bias相同而完全相同。
    output = linear(sample_input)

    print("\n" + "=" * 70)
    print("零偏置下的前向传播")
    print(f"sample_input.shape = {sample_input.shape}")
    print(f"output =\n{output}")

    # 为便于观察，构造一个简单标量loss=所有输出之和并执行反向传播。
    # 每个偏置分别加到2条样本对应输出上，因此d(loss)/d(bias)=2。
    demonstration_loss = output.sum()
    demonstration_loss.backward()

    print(f"反向传播后的bias.grad = {linear.bias.grad}")
    print(
        "说明：bias的初始值虽然是0，但requires_grad=True，"
        "所以训练时仍会产生梯度并被优化器更新。"
    )

    # ==================================================================
    # 第4步：演示普通权重全部为0时的对称性问题
    # ==================================================================

    # 创建另一个2输入、3输出的线性层，并且不使用偏置，以便只观察weight。
    symmetry_layer = nn.Linear(
        in_features=2,
        out_features=3,
        bias=False,
    )

    # 这是教学反例：把三个神经元的所有权重都初始化为0。
    nn.init.zeros_(symmetry_layer.weight)

    symmetry_input = torch.tensor([[1.0, 2.0]])
    symmetry_output = symmetry_layer(symmetry_input)

    print("\n" + "=" * 70)
    print("全零weight的对称性演示")
    print(f"初始weight =\n{symmetry_layer.weight}")
    print(f"输入        = {symmetry_input}")
    print(f"三个神经元输出 = {symmetry_output}")

    # 三个神经元参数完全相同，所以输出全都相同（这里都是0）。
    # 对三个输出求和后反向传播，每个神经元也会得到完全相同的梯度。
    symmetry_loss = symmetry_output.sum()
    symmetry_loss.backward()

    print(f"三个神经元的weight.grad =\n{symmetry_layer.weight.grad}")

    # 手动执行一次梯度下降，观察更新后三行权重仍完全相同。
    learning_rate = 0.1
    with torch.no_grad():
        symmetry_layer.weight -= (
            learning_rate * symmetry_layer.weight.grad
        )

    rows_still_equal = bool(
        torch.equal(
            symmetry_layer.weight[0],
            symmetry_layer.weight[1],
        )
        and torch.equal(
            symmetry_layer.weight[1],
            symmetry_layer.weight[2],
        )
    )

    print(f"一次梯度下降后的weight =\n{symmetry_layer.weight}")
    print(f"三行权重是否仍完全相同 = {rows_still_equal}")
    print(
        "这说明：如果相同神经元从完全相同的权重开始，并接收相同梯度，"
        "它们更新后仍然相同，无法学习不同特征。"
    )

    # ==================================================================
    # 第5步：常见初始化组合
    # ==================================================================

    # 普通全连接网络常见写法：
    #
    #     nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
    #     nn.init.zeros_(linear.bias)
    #
    # 或者使用Xavier初始化weight：
    #
    #     nn.init.xavier_uniform_(linear.weight)
    #     nn.init.zeros_(linear.bias)
    #
    # 这样weight通过随机值打破神经元对称性，bias则从中性的0开始学习。


def demo04() -> None:
    """演示全1初始化，以及为什么普通层权重通常不应全部初始化为1。

    ``nn.init.ones_(tensor)`` 会把张量中的所有元素原地设置为1。
    它和 ``zeros_`` 一样是确定性常量初始化，不是随机分布。

    对普通全连接隐藏层来说，把所有weight初始化为1通常存在两个问题：

    1. 同一层神经元从完全相同的权重开始，可能得到相同输出与相同梯度，
       更新后仍然相同，无法学习不同特征——这仍然是对称性问题。
    2. 一个神经元会把所有输入直接相加。如果输入维度较多或数值较大，
       输出及后续激活可能过大，导致Sigmoid/Tanh进入饱和区或造成数值不稳。

    全1初始化并非完全无用，它常用于具有明确数学含义的特殊参数，例如：

    * BatchNorm等归一化层的缩放参数gamma通常从1开始；
    * 某些门控、缩放因子、掩码或测试张量需要从1开始；
    * 研究代码中有时会有意让某些特定参数初始为1。

    但普通Linear/Conv的weight应优先使用Xavier、Kaiming等随机初始化。
    bias通常从0开始；把bias设为1会给每个输出额外增加固定偏移。
    """

    # 固定种子只是为了复现nn.Linear创建时的默认随机参数。
    # ones_本身没有随机性，无论随机种子是什么都会生成全1。
    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建线性层并观察默认参数
    # ==================================================================

    # 线性层执行：output = input @ weight.T + bias。
    #
    #     weight.shape=(3,5)，3个输出神经元各有5个权重
    #     bias.shape=(3,)，每个输出神经元各有一个偏置
    linear = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    print("=" * 70)
    print("nn.Linear创建后的默认随机参数（即将被ones_覆盖）")
    print(f"默认weight =\n{linear.weight}")
    print(f"默认bias   =\n{linear.bias}")

    # 保存初始化前对象身份和底层地址，用于验证ones_是否原地操作。
    weight_id_before = id(linear.weight)
    bias_id_before = id(linear.bias)
    weight_data_ptr_before = linear.weight.data_ptr()
    bias_data_ptr_before = linear.bias.data_ptr()

    # ==================================================================
    # 第2步：使用ones_原地覆盖weight和bias
    # ==================================================================

    # ones_末尾的下划线表示原地操作：直接修改原Parameter的数据，不创建
    # 新Parameter，也不改变requires_grad。nn.init函数不会把初始化记录到
    # autograd计算图中。
    returned_weight = nn.init.ones_(linear.weight)
    returned_bias = nn.init.ones_(linear.bias)

    print("\n" + "=" * 70)
    print("执行ones_之后的参数")
    print(f"weight =\n{linear.weight}")
    print(f"bias   =\n{linear.bias}")

    print("\n全1初始化检查：")
    print(
        "weight是否全部为1                  = "
        f"{bool((linear.weight == 1).all().item())}"
    )
    print(
        "bias是否全部为1                    = "
        f"{bool((linear.bias == 1).all().item())}"
    )
    print(
        "weight对象id是否不变               = "
        f"{weight_id_before == id(linear.weight)}"
    )
    print(
        "bias对象id是否不变                 = "
        f"{bias_id_before == id(linear.bias)}"
    )
    print(
        "weight底层地址是否不变             = "
        f"{weight_data_ptr_before == linear.weight.data_ptr()}"
    )
    print(
        "bias底层地址是否不变               = "
        f"{bias_data_ptr_before == linear.bias.data_ptr()}"
    )
    print(
        "ones_返回weight是否共享同一数据地址 = "
        f"{returned_weight.data_ptr() == linear.weight.data_ptr()}"
    )
    print(
        "ones_返回bias是否共享同一数据地址   = "
        f"{returned_bias.data_ptr() == linear.bias.data_ptr()}"
    )
    print(f"weight.requires_grad             = {linear.weight.requires_grad}")
    print(f"bias.requires_grad               = {linear.bias.requires_grad}")

    # ==================================================================
    # 第3步：计算全1参数下的前向传播
    # ==================================================================

    # 准备一条5特征样本。特征和为：1+2+3+4+5=15。
    sample_input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # 每个神经元的5个weight都是1、bias也是1，因此每个输出都是：
    #
    #     1×1 + 2×1 + 3×1 + 4×1 + 5×1 + 1 = 16
    #
    # 三个神经元参数完全相同，所以输出也完全相同。
    output = linear(sample_input)

    print("\n" + "=" * 70)
    print("全1参数下的前向传播")
    print(f"sample_input = {sample_input}")
    print(f"输入特征之和 = {sample_input.sum().item():.1f}")
    print(f"output       = {output}")

    expected_output = sample_input.sum() + 1
    print(f"手算每个输出 = {expected_output.item():.1f}")
    print(
        "三个输出是否完全相同 = "
        f"{bool((output == output[0, 0]).all().item())}"
    )

    # 这里输入只有5维，输出已经达到16。如果输入有1000个均值约1的特征，
    # 全1权重可能产生约1000量级的线性输出，使Sigmoid/Tanh很容易饱和。

    # ==================================================================
    # 第4步：演示全1weight的对称性问题
    # ==================================================================

    # 对三个输出求和形成标量，用来执行一次演示性反向传播。
    demonstration_loss = output.sum()
    demonstration_loss.backward()

    print("\n反向传播结果：")
    print(f"weight.grad =\n{linear.weight.grad}")
    print(f"bias.grad   = {linear.bias.grad}")

    # 对于loss=sum(output)：
    #
    #     每行weight.grad都等于输入[1,2,3,4,5]
    #     每个bias.grad都等于1
    #
    # 因为三个神经元起点和梯度相同，使用相同学习率更新后仍然相同。
    learning_rate = 0.1
    with torch.no_grad():
        linear.weight -= learning_rate * linear.weight.grad
        linear.bias -= learning_rate * linear.bias.grad

    weight_rows_still_equal = bool(
        torch.equal(linear.weight[0], linear.weight[1])
        and torch.equal(linear.weight[1], linear.weight[2])
    )
    bias_values_still_equal = bool(
        (linear.bias == linear.bias[0]).all().item()
    )

    print("\n一次梯度下降后的参数：")
    print(f"weight =\n{linear.weight}")
    print(f"bias   = {linear.bias}")
    print(f"三行weight是否仍完全相同 = {weight_rows_still_equal}")
    print(f"三个bias是否仍完全相同   = {bias_values_still_equal}")

    # 这个简化loss有意让三个神经元接收相同梯度，以直观展示对称性。
    # 真实网络梯度结构更复杂，但让普通层所有神经元从相同weight开始仍是
    # 不必要且危险的做法；随机初始化的核心价值之一就是打破这种对称性。

    # ==================================================================
    # 第5步：ones_与constant_的关系和适用场景
    # ==================================================================

    # ones_(tensor)等价于把常量1填入整个张量：
    #
    #     nn.init.ones_(tensor)
    #     nn.init.constant_(tensor, 1.0)
    #
    # zeros_(tensor)同理等价于constant_(tensor, 0.0)。
    # constant_可以填入任意指定常量，但普通层weight仍不应全部使用相同常量。
    #
    # 一个典型合理用途是归一化层的缩放参数：
    batch_norm = nn.BatchNorm1d(num_features=3)
    nn.init.ones_(batch_norm.weight)
    nn.init.zeros_(batch_norm.bias)

    print("\nBatchNorm常见初始化示例：")
    print(f"缩放参数gamma(weight) = {batch_norm.weight}")
    print(f"平移参数beta(bias)    = {batch_norm.bias}")

    # gamma=1、beta=0意味着归一化层初始先执行单位缩放和零平移：
    #
    #     output = normalized_input × 1 + 0
    #
    # 这里的全1有明确数学意义，因此是ones_的合理使用场景。

def demo05() -> None:
    """演示constant_常量初始化，以及大常量初始化的风险与合理用途。

    ``nn.init.constant_(tensor, value)`` 会把张量中的每个元素原地设置成
    指定常量value。它没有随机性，每次运行都得到完全相同的结果。

    zeros_和ones_可以看作constant_的特殊形式：

        nn.init.zeros_(tensor)         等价于 constant_(tensor, 0)
        nn.init.ones_(tensor)          等价于 constant_(tensor, 1)
        nn.init.constant_(tensor, 666) 把所有元素设为666

    普通隐藏层weight通常不应全部设置为相同常量：

    * 多个神经元可能从相同参数出发，产生相同输出和梯度，存在对称性问题；
    * 常量绝对值很大时，线性输出会非常大；
    * 巨大输出会使Sigmoid/Tanh进入饱和区，导数接近0，导致梯度消失；
    * 即使ReLU正半轴导数为1，巨大激活也可能导致后续数值和梯度不稳定。

    constant_的合理用途通常是“某个参数有明确初始含义”，例如零偏置、
    BatchNorm的gamma=1/beta=0，或者根据类别先验设置分类输出层偏置。
    """

    # 固定随机种子只影响nn.Linear创建时的默认随机参数。
    # constant_本身是确定性操作，不受随机种子影响。
    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建线性层并查看默认参数
    # ==================================================================

    # 线性层公式：output = input @ weight.T + bias。
    #
    #     weight.shape=(3,5)，共有15个权重
    #     bias.shape=(3,)，共有3个偏置
    linear = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    print("=" * 70)
    print("nn.Linear创建后的默认随机参数（即将被constant_覆盖）")
    print(f"默认weight =\n{linear.weight}")
    print(f"默认bias   =\n{linear.bias}")

    # 记录Parameter对象身份和底层数据地址，验证constant_是原地操作。
    weight_id_before = id(linear.weight)
    bias_id_before = id(linear.bias)
    weight_data_ptr_before = linear.weight.data_ptr()
    bias_data_ptr_before = linear.bias.data_ptr()

    # ==================================================================
    # 第2步：使用指定常量覆盖weight和bias
    # ==================================================================

    weight_constant = 666.0
    bias_constant = 999.0

    # constant_末尾的下划线表示原地修改已有Parameter数据，不会创建新参数，
    # 也不会把初始化操作记录到autograd计算图中。
    returned_weight = nn.init.constant_(
        linear.weight,
        weight_constant,
    )
    returned_bias = nn.init.constant_(
        linear.bias,
        bias_constant,
    )

    print("\n" + "=" * 70)
    print("执行constant_之后的参数")
    print(f"weight =\n{linear.weight}")
    print(f"bias   =\n{linear.bias}")

    print("\n常量初始化检查：")
    print(
        f"weight是否全部为{weight_constant:.1f} = "
        f"{bool((linear.weight == weight_constant).all().item())}"
    )
    print(
        f"bias是否全部为{bias_constant:.1f}   = "
        f"{bool((linear.bias == bias_constant).all().item())}"
    )
    print(
        "weight对象id是否不变             = "
        f"{weight_id_before == id(linear.weight)}"
    )
    print(
        "bias对象id是否不变               = "
        f"{bias_id_before == id(linear.bias)}"
    )
    print(
        "weight底层地址是否不变           = "
        f"{weight_data_ptr_before == linear.weight.data_ptr()}"
    )
    print(
        "bias底层地址是否不变             = "
        f"{bias_data_ptr_before == linear.bias.data_ptr()}"
    )
    print(
        "constant_返回weight是否共享地址  = "
        f"{returned_weight.data_ptr() == linear.weight.data_ptr()}"
    )
    print(
        "constant_返回bias是否共享地址    = "
        f"{returned_bias.data_ptr() == linear.bias.data_ptr()}"
    )
    print(f"weight.requires_grad           = {linear.weight.requires_grad}")
    print(f"bias.requires_grad             = {linear.bias.requires_grad}")

    # ==================================================================
    # 第3步：观察大常量产生的线性输出
    # ==================================================================

    # 使用一条5个特征全为1的样本。
    sample_input = torch.ones(1, 5)

    # 每个输出神经元的手算结果：
    #
    #     5个输入 × 666 + 偏置999
    #     = 5×666 + 999
    #     = 4329
    #
    # 三个神经元参数完全相同，所以三个输出也相同。
    output = linear(sample_input)
    expected_output = 5 * weight_constant + bias_constant

    print("\n" + "=" * 70)
    print("大常量参数下的前向传播")
    print(f"sample_input = {sample_input}")
    print(f"output       = {output}")
    print(f"手算每个输出 = {expected_output:.1f}")
    print(
        "三个输出是否完全相同 = "
        f"{bool((output == output[0, 0]).all().item())}"
    )

    # ==================================================================
    # 第4步：观察大输出对激活函数的影响
    # ==================================================================

    # 对4329这样的巨大正数：
    #
    #     sigmoid(4329)在float32中已经饱和为1
    #     tanh(4329)也已经饱和为1
    #     relu(4329)=4329，虽然正半轴导数为1，但数值仍然很大
    sigmoid_output = torch.sigmoid(output)
    tanh_output = torch.tanh(output)
    relu_output = torch.relu(output)

    # 使用激活值直接计算理论导数：
    #
    #     sigmoid'(x)=sigmoid(x)*(1-sigmoid(x))
    #     tanh'(x)=1-tanh(x)²
    sigmoid_gradient = sigmoid_output * (1 - sigmoid_output)
    tanh_gradient = 1 - tanh_output**2

    print("\n大线性输出经过激活函数：")
    print(f"sigmoid(output)      = {sigmoid_output}")
    print(f"sigmoid导数          = {sigmoid_gradient}")
    print(f"tanh(output)         = {tanh_output}")
    print(f"tanh导数             = {tanh_gradient}")
    print(f"relu(output)         = {relu_output}")
    print(
        "说明：Sigmoid/Tanh已经饱和且导数为0，梯度难以继续向前传播；"
        "ReLU没有正半轴饱和，但巨大激活仍会传给后续网络层。"
    )

    # ==================================================================
    # 第5步：演示相同常量weight的对称性
    # ==================================================================

    # 对三个输出求和并反向传播。三个神经元使用相同参数、接收相同上游梯度，
    # 因此weight.grad的三行完全相同，bias.grad的三个值也完全相同。
    demonstration_loss = output.sum()
    demonstration_loss.backward()

    print("\n反向传播结果：")
    print(f"weight.grad =\n{linear.weight.grad}")
    print(f"bias.grad   = {linear.bias.grad}")

    learning_rate = 0.1
    with torch.no_grad():
        linear.weight -= learning_rate * linear.weight.grad
        linear.bias -= learning_rate * linear.bias.grad

    weight_rows_still_equal = bool(
        torch.equal(linear.weight[0], linear.weight[1])
        and torch.equal(linear.weight[1], linear.weight[2])
    )

    print("\n一次梯度下降后的参数：")
    print(f"weight =\n{linear.weight}")
    print(f"bias   = {linear.bias}")
    print(f"三行weight是否仍完全相同 = {weight_rows_still_equal}")

    # ==================================================================
    # 第6步：constant_的合理用途——根据类别先验初始化输出偏置
    # ==================================================================

    # 假设一个二分类任务中，正样本先验比例约为10%。如果模型初始权重为0、
    # 偏置也为0，则初始sigmoid概率是0.5，与真实比例相差很大。
    # 可以把输出偏置初始化为先验概率p的logit：
    #
    #     bias = log(p / (1-p))
    #
    # 这样在输入贡献为0时，sigmoid(bias)会等于p。
    positive_prior = torch.tensor(0.10)
    prior_bias_value = torch.logit(positive_prior).item()

    binary_output_layer = nn.Linear(
        in_features=5,
        out_features=1,
        bias=True,
    )

    # 这里有明确数学目的，所以使用常量初始化是合理的。
    nn.init.zeros_(binary_output_layer.weight)
    nn.init.constant_(binary_output_layer.bias, prior_bias_value)

    zero_input = torch.zeros(1, 5)
    initial_logit = binary_output_layer(zero_input)
    initial_probability = torch.sigmoid(initial_logit)

    print("\n" + "=" * 70)
    print("根据二分类先验设置输出层偏置")
    print(f"正样本先验概率p         = {positive_prior.item():.4f}")
    print(f"对应logit偏置           = {prior_bias_value:.6f}")
    print(f"零输入时模型logit       = {initial_logit.item():.6f}")
    print(f"零输入时sigmoid概率     = {initial_probability.item():.6f}")

    # 得到的初始概率约为0.1，说明constant_本身没有好坏，关键是常量是否具有
    # 明确的模型意义。随意给普通weight填666通常不合理；根据先验设置输出
    # bias则可能帮助类别不平衡任务从更合理的初始预测开始。

def demo06() -> None:
    """演示Kaiming正态/均匀初始化及其与ReLU的关系。

    Kaiming初始化也叫He初始化，主要用于ReLU及其变体。它的目标是根据每层
    输入/输出连接数量自动调整权重尺度，让信号和梯度经过多层网络时不容易
    快速变成0（梯度消失），也不容易越来越大（激活或梯度爆炸）。

    对ReLU、``mode="fan_in"``，增益gain为 ``sqrt(2)``：

        Kaiming normal:
            weight ~ N(0, std²)
            std = gain / sqrt(fan_in) = sqrt(2 / fan_in)

        Kaiming uniform:
            weight ~ U(-bound, bound)
            bound = sqrt(3) * std = sqrt(6 / fan_in)

    ``fan_in`` 表示每个输出神经元接收多少输入连接，主要保持前向传播的
    方差；``fan_out`` 表示每个输入连接到多少输出，主要保持反向梯度方差。

    技术上，Kaiming并不是“只能初始化名字叫weight的张量”，而是要求张量
    至少二维，以便计算fan_in和fan_out。普通bias通常是一维张量，无法计算
    这两个值，所以不适用Kaiming，通常使用zeros_初始化。
    """

    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建两个独立线性层
    # ==================================================================

    # 使用两个层分别演示normal和uniform非常重要。如果对同一个weight连续
    # 调用两个初始化函数，后一次会原地覆盖前一次，模型最终只保留后一次结果。
    normal_layer = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )
    uniform_layer = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    print("\n" + "=" * 70)
    print("两个nn.Linear创建后的默认参数")
    print(f"normal_layer默认weight =\n{normal_layer.weight}")
    print(f"uniform_layer默认weight =\n{uniform_layer.weight}")

    # 对nn.Linear，weight形状为(out_features, in_features)=(3,5)。
    # 因此每个输出神经元接收5个输入，fan_in=5；共有3个输出，fan_out=3。
    fan_in = normal_layer.in_features
    fan_out = normal_layer.out_features

    print("\nLinear层的fan信息：")
    print(f"weight.shape = {normal_layer.weight.shape}")
    print(f"fan_in       = {fan_in}")
    print(f"fan_out      = {fan_out}")

    # ==================================================================
    # 第2步：计算ReLU对应的理论尺度
    # ==================================================================

    # calculate_gain根据激活函数返回增益。ReLU的gain=sqrt(2)。
    gain = nn.init.calculate_gain("relu")

    # mode="fan_in"是Kaiming函数的默认模式，本例显式写出。
    theoretical_std = gain / (fan_in**0.5)
    theoretical_variance = theoretical_std**2
    theoretical_bound = (3**0.5) * theoretical_std

    print("\nReLU + mode='fan_in'的理论初始化尺度：")
    print(f"gain                 = {gain:.10f}")
    print(f"normal理论均值       = 0")
    print(f"normal理论标准差     = {theoretical_std:.10f}")
    print(f"normal理论方差       = {theoretical_variance:.10f}")
    print(
        "uniform理论范围       = "
        f"[-{theoretical_bound:.10f}, {theoretical_bound:.10f}]"
    )

    # ==================================================================
    # 第3步：Kaiming正态初始化
    # ==================================================================

    # nonlinearity="relu"告诉初始化器使用ReLU的gain。
    # mode="fan_in"让初始化尺度依据输入连接数，优先稳定前向传播。
    # 初始化函数末尾“_”表示直接原地覆盖normal_layer.weight。
    returned_normal_weight = nn.init.kaiming_normal_(
        normal_layer.weight,
        mode="fan_in",
        nonlinearity="relu",
    )

    # bias不使用Kaiming，通常从0开始。
    nn.init.zeros_(normal_layer.bias)

    print("\n" + "=" * 70)
    print("Kaiming正态初始化结果")
    print(f"weight =\n{normal_layer.weight}")
    print(f"bias   = {normal_layer.bias}")
    print(
        "返回值是否与weight共享数据地址 = "
        f"{returned_normal_weight.data_ptr() == normal_layer.weight.data_ptr()}"
    )
    print(f"weight.requires_grad          = {normal_layer.weight.requires_grad}")

    normal_values = normal_layer.weight.detach()
    print(f"15个weight的实际均值          = {normal_values.mean().item():.10f}")
    print(
        "15个weight的实际标准差        = "
        f"{normal_values.std(unbiased=False).item():.10f}"
    )

    # 当前只有15个权重，样本统计不一定接近理论值，不能据此判断初始化错误。

    # ==================================================================
    # 第4步：Kaiming均匀初始化
    # ==================================================================

    returned_uniform_weight = nn.init.kaiming_uniform_(
        uniform_layer.weight,
        mode="fan_in",
        nonlinearity="relu",
    )
    nn.init.zeros_(uniform_layer.bias)

    print("\n" + "=" * 70)
    print("Kaiming均匀初始化结果")
    print(f"weight =\n{uniform_layer.weight}")
    print(f"bias   = {uniform_layer.bias}")
    print(
        "返回值是否与weight共享数据地址 = "
        f"{returned_uniform_weight.data_ptr() == uniform_layer.weight.data_ptr()}"
    )

    uniform_values = uniform_layer.weight.detach()
    all_uniform_values_in_range = bool(
        (
            (uniform_values >= -theoretical_bound)
            & (uniform_values <= theoretical_bound)
        )
        .all()
        .item()
    )

    print(f"实际最小值                    = {uniform_values.min().item():.10f}")
    print(f"实际最大值                    = {uniform_values.max().item():.10f}")
    print(f"所有值是否位于理论范围内      = {all_uniform_values_in_range}")
    print(f"15个weight的实际均值          = {uniform_values.mean().item():.10f}")
    print(
        "15个weight的实际标准差        = "
        f"{uniform_values.std(unbiased=False).item():.10f}"
    )

    # ==================================================================
    # 第5步：用大样本验证两种Kaiming分布的理论统计
    # ==================================================================

    # 创建形状(100000,5)的二维张量。对于mode="fan_in"，其fan_in仍是5，
    # 因此理论标准差和上面Linear(5,3)相同。大量元素使统计更接近理论值。
    large_normal_sample = torch.empty(100_000, fan_in)
    large_uniform_sample = torch.empty(100_000, fan_in)

    nn.init.kaiming_normal_(
        large_normal_sample,
        mode="fan_in",
        nonlinearity="relu",
    )
    nn.init.kaiming_uniform_(
        large_uniform_sample,
        mode="fan_in",
        nonlinearity="relu",
    )

    large_normal_mean = large_normal_sample.mean().item()
    large_normal_std = large_normal_sample.std(unbiased=False).item()
    large_uniform_mean = large_uniform_sample.mean().item()
    large_uniform_std = large_uniform_sample.std(unbiased=False).item()

    print("\n" + "=" * 70)
    print("大样本统计验证")
    print(
        f"Kaiming normal：理论均值=0，实际均值={large_normal_mean:.10f}"
    )
    print(
        "Kaiming normal："
        f"理论std={theoretical_std:.10f}，实际std={large_normal_std:.10f}"
    )
    print(
        f"Kaiming uniform：理论均值=0，实际均值={large_uniform_mean:.10f}"
    )
    print(
        "Kaiming uniform："
        f"理论std={theoretical_std:.10f}，实际std={large_uniform_std:.10f}"
    )
    print(
        "Kaiming uniform实际范围 = "
        f"[{large_uniform_sample.min().item():.10f}, "
        f"{large_uniform_sample.max().item():.10f}]"
    )

    # ==================================================================
    # 第6步：为什么不能直接初始化一维bias
    # ==================================================================

    # Kaiming需要从张量形状计算fan_in/fan_out。bias.shape=(3,)只有一维，
    # 不包含“输入连接数×输出连接数”结构，因此会抛出ValueError。
    try:
        nn.init.kaiming_normal_(
            normal_layer.bias,
            mode="fan_in",
            nonlinearity="relu",
        )
    except ValueError as error:
        print("\n一维bias使用Kaiming时的预期错误：")
        print(f"{type(error).__name__}: {error}")

    # 上面的调用在计算fan时就失败，没有成功修改bias；这里再次明确清零。
    nn.init.zeros_(normal_layer.bias)

    # ==================================================================
    # 第7步：mode与激活函数参数如何选择
    # ==================================================================

    # 1. 普通前向网络通常使用mode="fan_in"，尽量保持前向激活方差。
    # 2. 如果更关注反向梯度方差，可以选择mode="fan_out"；此时公式中的
    #    fan_in会换成fan_out，本例fan_out=3，权重尺度会不同。
    # 3. ReLU应写nonlinearity="relu"。
    # 4. LeakyReLU应把真实负半轴斜率a传给初始化器，例如：
    #
    #       negative_slope = 0.01
    #       nn.init.kaiming_normal_(
    #           layer.weight,
    #           a=negative_slope,
    #           mode="fan_in",
    #           nonlinearity="leaky_relu",
    #       )
    #
    # 5. Sigmoid/Tanh通常更适合Xavier初始化，而不是机械套用Kaiming。
    #
    # 重要：PyTorch按weight用于“input @ weight.T”的方式计算fan。
    # nn.Linear.weight正是(out_features,in_features)，可以直接传入初始化器。
    # 如果自定义参数以“input @ weight”方式使用，可能需要传weight.T初始化，
    # 否则fan_in和fan_out的解释会颠倒。

def demo07() -> None:
    """演示Xavier正态/均匀初始化及其与Sigmoid、Tanh的关系。

    Xavier初始化也叫Glorot初始化。它同时考虑输入连接数 ``fan_in`` 和输出
    连接数 ``fan_out``，在前向激活方差与反向梯度方差之间取得折中，降低
    信号经过多层网络后迅速变小或不断放大的风险。

    Xavier normal：

        weight ~ N(0, std²)
        std = gain * sqrt(2 / (fan_in + fan_out))

    Xavier uniform：

        weight ~ U(-bound, bound)
        bound = gain * sqrt(6 / (fan_in + fan_out))

    ``gain`` 根据后续激活函数调整权重尺度。PyTorch常见增益：

        Linear/Identity：1
        Sigmoid：1
        Tanh：5/3
        ReLU：sqrt(2)

    Xavier常用于线性、Sigmoid、Tanh网络；ReLU系列通常优先使用Kaiming。
    即使使用Xavier，过深的Sigmoid网络仍可能因导数很小而梯度消失，初始化
    只能缓解问题，不能改变激活函数本身的数学性质。
    """

    torch.manual_seed(42)

    # ==================================================================
    # 第1步：创建两个独立线性层
    # ==================================================================

    # 一个层演示Xavier normal，另一个层演示Xavier uniform。
    # 如果对同一个weight连续调用两个初始化函数，后一次会覆盖前一次，最后
    # 无法同时观察两种初始化结果。原demo07正存在这个问题。
    normal_layer = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )
    uniform_layer = nn.Linear(
        in_features=5,
        out_features=3,
        bias=True,
    )

    print("\n" + "=" * 70)
    print("两个nn.Linear创建后的默认参数")
    print(f"normal_layer默认weight =\n{normal_layer.weight}")
    print(f"uniform_layer默认weight =\n{uniform_layer.weight}")

    # nn.Linear.weight按(out_features,in_features)保存，当前形状为(3,5)：
    #
    #     fan_in=5：每个输出神经元接收5个输入
    #     fan_out=3：每个输入连接到3个输出神经元
    fan_in = normal_layer.in_features
    fan_out = normal_layer.out_features

    print("\nLinear层的fan信息：")
    print(f"weight.shape = {normal_layer.weight.shape}")
    print(f"fan_in       = {fan_in}")
    print(f"fan_out      = {fan_out}")
    print(f"fan_in + fan_out = {fan_in + fan_out}")

    # ==================================================================
    # 第2步：选择激活函数并计算理论尺度
    # ==================================================================

    # 本例假定线性层后面连接Tanh，因此使用Tanh对应的gain=5/3。
    # 如果后面使用Sigmoid，应改成calculate_gain("sigmoid")，结果为1。
    target_nonlinearity = "tanh"
    gain = nn.init.calculate_gain(target_nonlinearity)

    theoretical_std = gain * (
        2 / (fan_in + fan_out)
    ) ** 0.5
    theoretical_variance = theoretical_std**2
    theoretical_bound = gain * (
        6 / (fan_in + fan_out)
    ) ** 0.5

    print(f"\n目标激活函数：{target_nonlinearity}")
    print(f"gain                     = {gain:.10f}")
    print(f"normal理论均值           = 0")
    print(f"normal理论标准差         = {theoretical_std:.10f}")
    print(f"normal理论方差           = {theoretical_variance:.10f}")
    print(
        "uniform理论范围           = "
        f"[-{theoretical_bound:.10f}, {theoretical_bound:.10f}]"
    )

    # 顺便打印其他常见激活函数的gain，帮助比较。
    print("\n常见激活函数gain对比：")
    print(f"linear  gain = {nn.init.calculate_gain('linear'):.10f}")
    print(f"sigmoid gain = {nn.init.calculate_gain('sigmoid'):.10f}")
    print(f"tanh    gain = {nn.init.calculate_gain('tanh'):.10f}")
    print(f"relu    gain = {nn.init.calculate_gain('relu'):.10f}")

    # ==================================================================
    # 第3步：Xavier正态初始化
    # ==================================================================

    # xavier_normal_使用均值0、标准差theoretical_std的正态分布原地覆盖
    # normal_layer.weight。gain必须与后续激活函数匹配。
    returned_normal_weight = nn.init.xavier_normal_(
        normal_layer.weight,
        gain=gain,
    )
    nn.init.zeros_(normal_layer.bias)

    print("\n" + "=" * 70)
    print("Xavier正态初始化结果")
    print(f"weight =\n{normal_layer.weight}")
    print(f"bias   = {normal_layer.bias}")
    print(
        "返回值是否与weight共享数据地址 = "
        f"{returned_normal_weight.data_ptr() == normal_layer.weight.data_ptr()}"
    )
    print(f"weight.requires_grad          = {normal_layer.weight.requires_grad}")

    normal_values = normal_layer.weight.detach()
    print(f"15个weight的实际均值          = {normal_values.mean().item():.10f}")
    print(
        "15个weight的实际标准差        = "
        f"{normal_values.std(unbiased=False).item():.10f}"
    )

    # 只有15个权重，实际统计与理论值差异较大是正常随机波动。

    # ==================================================================
    # 第4步：Xavier均匀初始化
    # ==================================================================

    returned_uniform_weight = nn.init.xavier_uniform_(
        uniform_layer.weight,
        gain=gain,
    )
    nn.init.zeros_(uniform_layer.bias)

    print("\n" + "=" * 70)
    print("Xavier均匀初始化结果")
    print(f"weight =\n{uniform_layer.weight}")
    print(f"bias   = {uniform_layer.bias}")
    print(
        "返回值是否与weight共享数据地址 = "
        f"{returned_uniform_weight.data_ptr() == uniform_layer.weight.data_ptr()}"
    )

    uniform_values = uniform_layer.weight.detach()
    all_uniform_values_in_range = bool(
        (
            (uniform_values >= -theoretical_bound)
            & (uniform_values <= theoretical_bound)
        )
        .all()
        .item()
    )

    print(f"实际最小值                    = {uniform_values.min().item():.10f}")
    print(f"实际最大值                    = {uniform_values.max().item():.10f}")
    print(f"所有值是否位于理论范围内      = {all_uniform_values_in_range}")
    print(f"15个weight的实际均值          = {uniform_values.mean().item():.10f}")
    print(
        "15个weight的实际标准差        = "
        f"{uniform_values.std(unbiased=False).item():.10f}"
    )

    # ==================================================================
    # 第5步：用大样本验证Xavier计算出的分布参数
    # ==================================================================

    # Xavier同时依赖fan_in和fan_out，不能简单创建(100000,5)张量再调用
    # xavier_*验证，因为该张量的fan_out会变成100000，理论尺度也随之改变。
    # 因此这里先按当前Linear(5,3)公式算出std/bound，再用normal_/uniform_
    # 生成大量一维独立样本，验证这两个分布参数本身。
    sample_count = 500_000
    large_normal_sample = torch.empty(sample_count)
    large_uniform_sample = torch.empty(sample_count)

    nn.init.normal_(
        large_normal_sample,
        mean=0.0,
        std=theoretical_std,
    )
    nn.init.uniform_(
        large_uniform_sample,
        a=-theoretical_bound,
        b=theoretical_bound,
    )

    print("\n" + "=" * 70)
    print("大样本统计验证")
    print(
        "Xavier normal："
        f"理论mean=0，实际mean={large_normal_sample.mean().item():.10f}"
    )
    print(
        "Xavier normal："
        f"理论std={theoretical_std:.10f}，"
        f"实际std={large_normal_sample.std(unbiased=False).item():.10f}"
    )
    print(
        "Xavier uniform："
        f"理论mean=0，实际mean={large_uniform_sample.mean().item():.10f}"
    )
    print(
        "Xavier uniform："
        f"理论std={theoretical_std:.10f}，"
        f"实际std={large_uniform_sample.std(unbiased=False).item():.10f}"
    )
    print(
        "Xavier uniform实际范围 = "
        f"[{large_uniform_sample.min().item():.10f}, "
        f"{large_uniform_sample.max().item():.10f}]"
    )

    # ==================================================================
    # 第6步：为什么普通一维bias不使用Xavier
    # ==================================================================

    # 与Kaiming一样，Xavier需要至少二维的连接结构来计算fan_in和fan_out。
    # bias.shape=(3,)是一维张量，因此直接调用会抛出ValueError。
    try:
        nn.init.xavier_normal_(normal_layer.bias, gain=gain)
    except ValueError as error:
        print("\n一维bias使用Xavier时的预期错误：")
        print(f"{type(error).__name__}: {error}")

    nn.init.zeros_(normal_layer.bias)

    # ==================================================================
    # 第7步：Xavier与Kaiming如何选择
    # ==================================================================

    # Xavier：同时使用fan_in和fan_out，常配合Linear/Sigmoid/Tanh。
    # Kaiming：通常选择fan_in或fan_out，专门考虑ReLU系列的半轴截断特性。
    #
    # 示例：Tanh网络
    #
    #     gain = nn.init.calculate_gain("tanh")
    #     nn.init.xavier_uniform_(layer.weight, gain=gain)
    #     nn.init.zeros_(layer.bias)
    #
    # 示例：ReLU网络
    #
    #     nn.init.kaiming_uniform_(
    #         layer.weight,
    #         mode="fan_in",
    #         nonlinearity="relu",
    #     )
    #     nn.init.zeros_(layer.bias)
    #
    # 重要：PyTorch假设weight按“input @ weight.T”使用。nn.Linear.weight的
    # (out_features,in_features)布局符合该假设；自定义反向布局时需考虑转置。


def initialize_linear_by_activation(
    layer: nn.Linear,
    activation: str,
    negative_slope: float = 0.01,
) -> str:
    """根据线性层后面的激活函数初始化weight，并把bias初始化为0。

    Args:
        layer: 需要初始化的 ``nn.Linear`` 层。
        activation: 后续激活函数名称，例如relu、leaky_relu、tanh、sigmoid。
        negative_slope: LeakyReLU负半轴斜率，仅在activation为leaky_relu时使用。

    Returns:
        实际使用的初始化方法说明，方便示例打印。

    这是一个教学辅助函数，展示“初始化方式应与后续激活函数匹配”。真实项目
    还要考虑卷积层、残差结构、归一化层、框架默认初始化以及已有预训练权重。
    """

    activation = activation.lower()

    if activation == "relu":
        # ReLU负半轴输出0，Kaiming会用sqrt(2)增益补偿方差损失。
        nn.init.kaiming_uniform_(
            layer.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        method = "Kaiming uniform（ReLU）"

    elif activation == "leaky_relu":
        # LeakyReLU负半轴仍保留negative_slope倍输入，因此必须把真实斜率
        # 传给参数a，否则初始化器使用的增益与实际激活函数不匹配。
        nn.init.kaiming_uniform_(
            layer.weight,
            a=negative_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        method = (
            "Kaiming uniform（LeakyReLU，"
            f"negative_slope={negative_slope}）"
        )

    elif activation == "tanh":
        # Tanh使用Xavier，并通过calculate_gain取得5/3增益。
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        method = f"Xavier uniform（Tanh，gain={gain:.6f}）"

    elif activation == "sigmoid":
        # Sigmoid对应gain=1。Xavier有助于控制线性输出尺度，减少一开始就
        # 大量落入Sigmoid饱和区的风险，但无法彻底解决深层梯度消失。
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        method = f"Xavier uniform（Sigmoid，gain={gain:.6f}）"

    elif activation in {"linear", "identity", "none"}:
        # 无激活/恒等激活使用gain=1的Xavier作为通用选择。
        gain = nn.init.calculate_gain("linear")
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        method = "Xavier uniform（Linear/Identity，gain=1）"

    else:
        raise ValueError(
            "不支持的activation："
            f"{activation!r}。可选relu、leaky_relu、tanh、sigmoid、linear。"
        )

    # 图片中的建议：偏置b使用全0初始化。
    # bias从0开始不会造成weight那样的神经元对称问题，因为随机weight已经
    # 打破了对称性；bias仍然requires_grad=True，训练后可以离开0。
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    return method


def demo08() -> None:
    """汇总图片中的初始化推荐，并给出可执行示例和适用边界。

    图片《使用推荐总结》的核心决策树：

        参数类型
        ├─ bias偏置
        │   └─ 全0初始化
        └─ weight权重
            ├─ 根据激活函数
            │   ├─ ReLU系列     → Kaiming初始化
            │   └─ Tanh/Sigmoid → Xavier初始化
            └─ 根据网络层数
                ├─ 浅层网络（层数<=10）
                │   ├─ 均匀分布初始化
                │   ├─ 正态分布初始化
                │   ├─ Kaiming初始化
                │   └─ Xavier初始化
                └─ 深层网络（层数>10）
                    ├─ Kaiming初始化
                    └─ Xavier初始化

    需要补充的准确理解：

    1. “层数<=10”和“层数>10”是便于入门的经验划分，不是数学定理；网络
       是否稳定还受宽度、激活函数、归一化、残差连接、优化器等因素影响。
    2. 激活函数通常比单纯层数更直接地决定初始化公式：ReLU优先Kaiming，
       Tanh/Sigmoid优先Xavier。即使是浅层网络，也建议优先遵循这个匹配。
    3. “浅层可用普通均匀/正态”不表示可以随意使用U(0,1)或N(0,1)。如果
       权重尺度不结合fan_in/fan_out，仍可能使输出过大、激活饱和或训练不稳。
    4. 深层网络优先Xavier/Kaiming只是基础；现代深网通常还配合BatchNorm、
       LayerNorm、残差连接、合理优化器，有时还使用框架或论文指定初始化。
    5. 如果加载预训练模型，通常不要重新初始化已有权重，否则会破坏训练成果。
    """

    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("神经网络参数初始化推荐总结")
    print("bias：通常使用zeros_初始化")
    print("ReLU/LeakyReLU：通常使用Kaiming初始化")
    print("Tanh/Sigmoid：通常使用Xavier初始化")
    print("深层网络：优先使用与激活函数匹配的Xavier/Kaiming，并结合归一化/残差")

    # ==================================================================
    # 第1步：为不同激活函数创建相同形状的线性层
    # ==================================================================

    # 使用128个输入、64个输出。每层有8192个weight，样本数量比前面(3,5)
    # 更大，实际均值和标准差通常更接近初始化公式的理论值。
    configurations = [
        ("relu", None),
        ("leaky_relu", 0.01),
        ("tanh", None),
        ("sigmoid", None),
    ]

    initialized_layers: dict[str, nn.Linear] = {}

    for activation, slope in configurations:
        layer = nn.Linear(
            in_features=128,
            out_features=64,
            bias=True,
        )

        if slope is None:
            method = initialize_linear_by_activation(
                layer,
                activation,
            )
        else:
            method = initialize_linear_by_activation(
                layer,
                activation,
                negative_slope=slope,
            )

        initialized_layers[activation] = layer
        values = layer.weight.detach()

        print("\n" + "-" * 70)
        print(f"后续激活函数：{activation}")
        print(f"初始化方法：{method}")
        print(f"weight.shape = {layer.weight.shape}")
        print(f"weight均值  = {values.mean().item():.10f}")
        print(
            "weight标准差 = "
            f"{values.std(unbiased=False).item():.10f}"
        )
        print(
            "bias是否全部为0 = "
            f"{bool((layer.bias == 0).all().item())}"
        )

    # ==================================================================
    # 第2步：验证bias从0开始仍然可以学习
    # ==================================================================

    relu_layer = initialized_layers["relu"]
    sample_input = torch.randn(4, 128)
    sample_output = torch.relu(relu_layer(sample_input))
    demonstration_loss = sample_output.sum()
    demonstration_loss.backward()

    print("\n" + "=" * 70)
    print("零偏置仍能学习的验证（ReLU层）")
    print(f"bias初始值是否全0 = {bool((relu_layer.bias == 0).all().item())}")
    print(f"bias.grad          = {relu_layer.bias.grad}")
    print(
        "bias.grad中非零元素数量 = "
        f"{torch.count_nonzero(relu_layer.bias.grad).item()}"
    )
    print(
        "结论：zeros_只设置初值，不会冻结bias；只要requires_grad=True，"
        "反向传播仍会计算梯度，优化器仍会更新它。"
    )

    # ==================================================================
    # 第3步：把图片经验整理为实际选择顺序
    # ==================================================================

    print("\n" + "=" * 70)
    print("实际项目中的推荐判断顺序")
    print("1. 先检查是否加载预训练权重；若是，通常不要重新初始化。")
    print("2. 查看层后面的激活函数：ReLU系列→Kaiming，Tanh/Sigmoid→Xavier。")
    print("3. bias没有特殊先验时通常设0；有明确先验时可按任务设置。")
    print("4. 深层网络再检查归一化、残差结构以及论文/架构指定的初始化方案。")
    print("5. 最后通过激活值、梯度范数和训练曲线验证初始化是否真的合适。")


if __name__ == "__main__":
    # demo01()
    # demo02()
    # demo03()
    # demo04()
    # demo05()
    # demo06()
    # demo07()
    demo08()
