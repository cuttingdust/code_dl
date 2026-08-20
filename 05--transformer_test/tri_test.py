import torch
import numpy as np

# 使用PyTorch实现掩码。【掌握】
"""
    总结：
        1- 上三角掩码：右上角的值为1，左下角的值为0
        2- 下三角掩码：右上角的值为0，左下角的值为1
        3- 相同的规律：对角线的时候diagonal为0；如果是正数，并且越来越大，那么往右上角移动；反之往左下角移动；diagonal的取值超过范围不会报错
        4- 【掌握】因果掩码的作用：防止解码器当前时间步偷看未来词，避免训练时发生答案泄漏
        5- 本项目约定1表示允许关注、0表示屏蔽，因此因果掩码使用：torch.tril(t, diagonal=0)
           如果某套代码约定True表示要屏蔽，才会使用上三角矩阵；必须结合masked_fill的判断条件理解
"""


def torch_mask():
    print("-" * 50)
    # 创建初始张量
    t = torch.ones(size=(5, 5))
    print(f"原始张量:{t}")

    # 进行掩码操作
    # 上三角掩码
    print("-" * 50)
    u_0_mask = torch.triu(t, diagonal=0)
    print(f"上三角掩码0:{u_0_mask}")

    print("-" * 50)
    u_1_mask = torch.triu(t, diagonal=1)
    print(f"上三角掩码1：{u_1_mask}")

    print("-" * 50)
    u__1_mask_ = torch.triu(t, diagonal=-1)
    print(f"上三角掩码-1：{u__1_mask_}")

    print("-" * 50)
    u_n_mask = torch.triu(t, diagonal=1000)
    print(f"上三角掩码n：{u_n_mask}")

    # 下三角掩码
    print("#" * 100)
    l_0_mask = torch.tril(t, diagonal=0)
    print(f"下三角掩码0：{l_0_mask}")

    print("-" * 50)
    l_1_mask = torch.tril(t, diagonal=1)
    print(f"下三角掩码1：{l_1_mask}")

    print("-" * 50)
    l__1_mask = torch.tril(t, diagonal=-1)
    print(f"下三角掩码-1：{l__1_mask}")


def np_mask():
    print("-" * 50)
    # 1- 准备数据
    arr = np.ones(shape=(5, 5))

    # 2- 上三角掩码
    # k和PyTorch中diagonal参数的作用完全一样
    u_result = np.triu(arr, k=0)
    print(f"上三角掩码0：\n{u_result}")

    print("-" * 50)
    u_result = np.triu(arr, k=1)
    print(f"上三角掩码1：\n{u_result}")

    print("-" * 50)
    u_result = np.triu(arr, k=-1)
    print(f"上三角掩码-1：\n{u_result}")

    print("#" * 100)

    # 3- 下三角掩码
    l_result = 1 - u_result
    print(f"下三角掩码：\n{l_result}")

    print("-" * 50)
    l_result = np.tril(arr, k=0)
    print(f"下三角掩码：\n{l_result}")

    print("-" * 50)
    l_result = np.tril(arr, k=1)
    print(f"下三角掩码1：\n{l_result}")

    print("-" * 50)
    l_result = np.tril(arr, k=-1)
    print(f"下三角掩码-1：\n{l_result}")


def assert_demo(d_model, head):
    print("assert之前")
    assert d_model % head == 0
    print("assert之后")


def mask():
    # 一条长度为4的目标句子使用下三角因果掩码。
    # [4,4] -> [1,1,4,4]，随后广播到2个批次、2个注意力头。
    causal_mask = torch.tril(torch.ones(size=(4, 4), dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    scores = torch.ones(size=(2, 2, 4, 4))
    scores = scores.masked_fill(~causal_mask, value=-1e9)

    print(f"掩码后的注意力分数：\n{scores}")
    print("-" * 50)
    print(f"因果掩码：\n{causal_mask}")


def k_multi_data():
    """
    return self.k * (data - mean)/(std+self.eps) + self.b
    演示 为什么 self.k * data 能够相乘？-> 广播机制
    k的形状是[512]，data形状[2,4,512]
    """

    k = torch.randint(low=1, high=5, size=(3,))
    data = torch.randint(low=1, high=5, size=(2, 4, 3))

    print(f"k -->\n{k}")
    print(f"data-->\n{data}")
    print(f"k * data-->\n{k * data}")


if __name__ == "__main__":
    # torch版的掩码
    # torch_mask()

    # np_mask()

    # assert_demo(512, 8)
    # assert_demo(512, 7)

    # mask()

    k_multi_data()
