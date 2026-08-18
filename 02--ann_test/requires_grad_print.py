"""演示 PyTorch 中梯度、对象身份、内存共享和数据转换。

本示例集中回答以下问题：

1. 设置 requires_grad=True 后，为什么 tensor.grad 一开始仍然是 None？
2. tensor.data、tensor.item() 和 tensor.numpy() 分别返回什么？
3. id() 不同是否意味着底层数据内存也不同？
4. detach() 和 clone() 有什么区别？
5. 如何得到与原张量完全独立的 Tensor 或 NumPy 数组？

核心结论：

    requires_grad=True        允许 autograd 跟踪运算，但不会立即产生梯度
    backward()                执行反向传播后，叶子张量的 .grad 才会有值
    detach()                  脱离计算图，但仍与原张量共享底层存储
    clone()                   复制底层数据，但单独使用时仍保留自动微分关系
    detach().clone()          既脱离计算图，又拥有独立底层存储
    detach().cpu().numpy()    转为 NumPy，通常仍与 CPU Tensor 共享存储
    ...numpy().copy()         得到与 Tensor 完全独立的 NumPy 数组
"""

import torch


def print_title(title: str) -> None:
    """打印分节标题，让不同实验的输出更容易观察。"""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


if __name__ == "__main__":
    # ==================================================================
    # 实验 1：requires_grad=True 不等于已经有梯度
    # ==================================================================
    print_title("1. requires_grad 与 grad")

    # t1 是我们直接创建的叶子张量。
    # requires_grad=True 仅表示：后续由 t1 参与的运算需要被 autograd 记录。
    # 此时还没有定义损失，也没有调用 backward()，所以 t1.grad 是 None。
    t1 = torch.tensor(20.0, dtype=torch.float32, requires_grad=True)

    print(f"t1                 = {t1}")
    print(f"t1.requires_grad   = {t1.requires_grad}")
    print(f"反向传播前 t1.grad = {t1.grad}")

    # 前向传播：loss=2t1²。此时 PyTorch 建立计算图，但仍未计算梯度。
    loss = 2 * t1**2
    print(f"loss               = {loss}")
    print(f"loss.grad_fn       = {loss.grad_fn}")
    print(f"backward 前 t1.grad = {t1.grad}")

    # 反向传播：d(2t1²)/dt1=4t1。t1=20，所以梯度为 80。
    # backward() 之后，叶子张量 t1 的梯度才保存到 t1.grad。
    loss.sum().backward()
    print(f"backward 后 t1.grad = {t1.grad}")

    # ==================================================================
    # 实验 2：.data、.item() 与对象类型
    # ==================================================================
    print_title("2. .data 与 .item()")

    # .data 返回另一个 Tensor 对象，它不需要梯度，但和 t1 共享底层存储。
    # .data 会绕过 autograd 的安全检查，修改它可能破坏计算图的一致性。
    # 这里仅用于观察；实际代码应优先使用 detach()，不要用 .data 更新参数。
    data_tensor = t1.data

    # .item() 只适用于单元素张量，返回一个普通 Python 数字。
    # Python 数字只是当前值的副本，不共享 Tensor 的底层存储。
    python_number = t1.item()

    print(f"t1.data                    = {data_tensor}")
    print(f"type(t1.data)              = {type(data_tensor)}")
    print(f"t1.data.requires_grad      = {data_tensor.requires_grad}")
    print(f"t1.item()                  = {python_number}")
    print(f"type(t1.item())            = {type(python_number)}")

    # ==================================================================
    # 实验 3：id() 与 data_ptr() 观察的是不同层次
    # ==================================================================
    print_title("3. id 与 data_ptr")

    detached = t1.detach()

    # id() 表示 Python 对象身份。t1 和 detached 是两个不同的 Tensor 对象，
    # 所以它们的 id 不同。
    print(f"id(t1)                     = {id(t1)}")
    print(f"id(detached)               = {id(detached)}")
    print(f"t1 is detached             = {t1 is detached}")

    # data_ptr() 表示张量第一个元素所在的底层数据地址。
    # detach() 不复制数据，所以两者的 data_ptr() 相同。
    print(f"t1.data_ptr()              = {t1.data_ptr()}")
    print(f"detached.data_ptr()        = {detached.data_ptr()}")
    print(
        "是否共享底层存储          = "
        f"{t1.data_ptr() == detached.data_ptr()}"
    )

    # 准确表述是：
    # “detach() 后对象的 id 不同，但底层数据存储地址相同。”
    # 不能只说“内存地址不同”，因为对象和数据存储属于两个不同层次。

    # ==================================================================
    # 实验 4：detach() 脱离计算图，但仍然共享数据
    # ==================================================================
    print_title("4. detach 仍然共享底层数据")

    source = torch.tensor([10.0, 20.0], requires_grad=True)
    detached_view = source.detach()

    print(f"修改前 source              = {source}")
    print(f"修改前 detached_view       = {detached_view}")
    print(f"对象是否相同               = {source is detached_view}")
    print(
        "底层地址是否相同           = "
        f"{source.data_ptr() == detached_view.data_ptr()}"
    )

    # detached_view 不需要梯度，可以执行原地修改。
    # 由于它和 source 共享底层存储，修改 detached_view 也会改变 source。
    # 因此 detach() 不是“复制数据”，只是让一个视图脱离计算图。
    detached_view.add_(1000)

    print(f"修改后 source              = {source}")
    print(f"修改后 detached_view       = {detached_view}")

    # ==================================================================
    # 实验 5：detach().clone() 得到完全独立的普通张量
    # ==================================================================
    print_title("5. detach().clone() 独立存储")

    original = torch.tensor([10.0, 20.0], requires_grad=True)

    # detach()：切断自动微分关系。
    # clone()：复制一份独立的底层数据。
    independent = original.detach().clone()

    print(f"original                   = {original}")
    print(f"independent                = {independent}")
    print(f"independent.requires_grad  = {independent.requires_grad}")
    print(f"independent.grad_fn        = {independent.grad_fn}")
    print(
        "底层地址是否相同           = "
        f"{original.data_ptr() == independent.data_ptr()}"
    )

    independent.add_(1000)

    # independent 有自己的存储，所以修改它不会影响 original。
    print(f"修改后 original           = {original}")
    print(f"修改后 independent        = {independent}")

    # ==================================================================
    # 实验 6：只使用 clone() 会复制数据，但不会切断计算图
    # ==================================================================
    print_title("6. 单独 clone() 仍保留梯度关系")

    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    connected_clone = parameter.clone()

    print(f"connected_clone            = {connected_clone}")
    print(f"requires_grad              = {connected_clone.requires_grad}")
    print(f"grad_fn                    = {connected_clone.grad_fn}")
    print(
        "底层地址是否相同           = "
        f"{parameter.data_ptr() == connected_clone.data_ptr()}"
    )

    # connected_clone 拥有独立存储，但仍在 parameter 的计算图中。
    # 对它计算的损失执行 backward()，梯度仍然能够传回 parameter。
    clone_loss = (connected_clone**2).sum()
    clone_loss.backward()
    print(f"parameter.grad             = {parameter.grad}")

    # ==================================================================
    # 实验 7：Tensor 转 NumPy，以及 NumPy 的内存共享
    # ==================================================================
    print_title("7. Tensor 与 NumPy")

    tensor_for_numpy = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # requires_grad=True 的张量不能直接调用 numpy()，因为 NumPy 不支持
    # PyTorch 的自动微分。需要先 detach()，GPU 张量还需要先 cpu()。
    # 这是兼容 CPU/GPU 张量的常用写法。
    numpy_shared = tensor_for_numpy.detach().cpu().numpy()

    print(f"numpy_shared               = {numpy_shared}")
    print(f"type(numpy_shared)         = {type(numpy_shared)}")
    print(f"numpy_shared.shape         = {numpy_shared.shape}")

    # CPU Tensor 转换出的 NumPy 数组通常与 Tensor 共享底层内存。
    # 修改 NumPy 数组，也会修改原 Tensor 的数据。
    numpy_shared[0] = 100.0
    print(f"修改 NumPy 后的数组         = {numpy_shared}")
    print(f"修改 NumPy 后的 Tensor      = {tensor_for_numpy}")

    # 如果希望 NumPy 数组完全独立，需要再调用 NumPy 的 copy()。
    numpy_independent = tensor_for_numpy.detach().cpu().numpy().copy()
    numpy_independent[1] = 999.0

    print(f"独立 NumPy 数组            = {numpy_independent}")
    print(f"原 Tensor 不受 copy 影响    = {tensor_for_numpy}")

    # ==================================================================
    # 最终对照总结
    # ==================================================================
    print_title("总结")
    print("t2 = t1                  ：同一个 Tensor 对象，共享存储和计算图")
    print("t2 = t1.detach()         ：不同对象，脱离计算图，但共享存储")
    print("t2 = t1.clone()          ：不同对象、独立存储，但保留梯度关系")
    print("t2 = t1.detach().clone() ：不同对象、独立存储，并脱离计算图")
    print("t1.detach().cpu().numpy()：NumPy 数组，通常与 CPU Tensor 共享存储")
    print("...numpy().copy()        ：与 Tensor 完全独立的 NumPy 数组")
