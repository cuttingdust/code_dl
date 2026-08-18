import torch

"""
    matmul和bmm总结
        1- 过程类似，都是对张量进行矩阵乘法的运算
        2- matmul是串行计算；bmm内部是并行计算
        3- bmm对两个张量的要求如下：
            3.1- 两个张量维度的第一个位置上的形状要完全相同，该位置表示的是batch_size
            3.2- 只能计算三维张量
        4- 为什么matmul能够对(1,3,4)和(10,4,5)能进行矩阵乘法。是因为内部有广播机制
            也就是会把(1,3,4)的形状扩大成(10,3,4)的形状
"""

if __name__ == "__main__":
    # 准备数据
    # mat_1 = torch.randn(size=(10, 3, 4))
    mat_1 = torch.randn(size=(1, 3, 4))
    mat_2 = torch.randn(size=(10, 4, 5))

    # matmul
    matmul_result = torch.matmul(mat_1, mat_2)
    print("matmul形状：", matmul_result.shape)  # 10,3,5

    # bmm
    bmm_result = torch.bmm(mat_1, mat_2)
    print("bmm形状：", bmm_result.shape)  # 10,3,5
