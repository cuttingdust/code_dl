import jieba
from itertools import chain

if __name__ == "__main__":
    # 解释set(chain(*map(lambda)))

    my_data = ["今天天气很好", "晚上吃什么"]

    result_1 = map(
        lambda line: jieba.lcut(line),
        my_data,
    )

    # [['今天天气', '很', '好'], ['晚上', '吃', '什么']]
    # print(list(result_1))
    # print(*result_1)

    # chain的作用类似list.extend，优点是非常节约内存（生成器和list的区别）
    # *表示解包处理
    result_2 = chain(*result_1)
    print(set(result_2))

    # 上面代码合并后的写法
    final_result = set(chain(*map(lambda line: jieba.lcut(line), my_data)))
    print(final_result)
