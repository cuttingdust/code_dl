# 演示：list( map(lambda x: len(x), train_data['sentence']))
def demo01():
    my_list = ["hello", "abc", "heimaaaa"]

    # 返回的实际是一个生成器对象。优点：节约内存
    # map的作用是：将my_list容器中的元素一个个遍历出来，作为参数传递给到前面的自定义函数或lambda表达式
    map_result = map(lambda x: len(x), my_list)
    # print(type(map_result))
    # print(map_result)

    # 取值方式一：推荐，转成list列表
    # 取值方式一：推荐，转成list列表
    # list_result = list(map_result)
    # 综合之后的代码版本
    list_result = list(map(lambda x: len(x), my_list))
    print(type(list_result))
    print(list_result)

    # 取值方式二：for循环
    # for x in list_result:
    #     print(x)
    #
    # # 取值方式三：next取值
    # print(next(map_result))
    # print(next(map_result))
    # print(next(map_result))


# 演示：list(chain(*map(lambda x: get_a_list(x) , p_train_data)))
import jieba
from itertools import chain


def demo02():
    sen_list = ["今天天气很好", "元旦放两天"]
    map_result = list(map(lambda line: jieba.lcut(line), sen_list))

    # for i in map_result:
    #     print(i)

    # print(*map_result)

    # 总结：chain的作用等价于List.extend，优点是节约内存
    chain_result = chain(*map_result)
    print(type(chain_result))
    print(chain_result)

    print(list(chain_result))

    # 最终版代码
    final_result = list(chain(*map(lambda line: jieba.lcut(line), sen_list)))
    print(final_result)


# 演示：zip函数
def demo03():
    list_1 = [11, 22, 33, 44, 55]
    list_2 = ["hello", "python", "java"]

    zip_result_1 = list(zip(list_1, list_2))
    print(type(zip_result_1))
    print(zip_result_1)

    print("-" * 30)

    list_1 = ["hello", "python", "java"]
    list_2 = [11, 22, 33, 44, 55]

    zip_result_2 = list(zip(list_1, list_2))
    print(zip_result_2)


import torch
import torch.nn as nn


def demo04():
    # 1- 准备线性求和结果数据
    data = torch.tensor([5.6, 8.7, -3.4])

    # 2- 调用LogSoftmax：线性求和结果 -> Softmax概率值 -> 以e为底的幂指数
    log_softmax = nn.LogSoftmax(dim=-1)
    log_softmax_result = log_softmax(data)
    print(log_softmax_result)

    # 3- 调用Softmax：线性求和结果 -> Softmax概率值
    softmax = nn.Softmax(dim=-1)
    softmax_result = softmax(data)
    print(softmax_result)

    # 4- 写代码验证
    print(torch.log(softmax_result))


if __name__ == "__main__":
    # demo01()

    # demo02()

    # demo03()

    # LogSoftmax和Softmax的区别
    demo04()
