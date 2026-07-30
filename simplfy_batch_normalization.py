import torch
from torch import nn

if __name__ == "__main__":
    # 准备输入数据
    """
    样本数据结构是4维的，每个位置上的参数含义解释：
        1：样本中有1张图片
        2：图片的颜色通道个数C
        3：图片的高度H
        4：图片的宽度W
    """
    input_2d = torch.randn(1, 2, 3, 4)

    # 批量归一化BN层
    """
        BatchNorm2d中参数解释： 
            num_features：在处理图片的时候，输入图片的通道数有几个
            eps：是一个小常数，为了防止分母为0
            affine：该值通常设置为True。表示神经网络内部自动的去学习得到公式中的γ和β系数


        类的作用解释：
            BatchNorm1d：主要用来处理文本内容。输入数据的形状是 N,num_features
                N：样本数据条数
                num_features：输入样本的特征个数

            BatchNorm2d：主要用来处理图片。输入数据的形状是 N,C,H,W。
                N：图片的张数
                C：图片的通道数。常见图片的通道数为3 RGB
                H：图片的高度
                W：图片的宽度

            BatchNorm3d：主要用来处理视频（视频是有多张图片组成的）。输入数据的形状是 N,C,D,H,W。
                N：图片的张数
                C：图片的通道数。常见图片的通道数为3 RGB
                D：是维度数
                H：图片的高度
                W：图片的宽度 
    """
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    output = bn2d(input_2d)
    print("公式中的γ", bn2d.weight)
    print("公式中的β", bn2d.bias)
