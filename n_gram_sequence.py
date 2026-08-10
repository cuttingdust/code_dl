import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing import sequence

"""
    句子长度规范的作用【掌握】：
        1- Dataloader(batch_size=1)，张量的形状不相同的时候只能设置为1。因为里面是使用stack对张量进行拼接
            stack的要求是张量形状完全相同
        2- 迁移学习中有现成的参数可以进行设置
"""


def demo01():
    # 1- 准备数据，两条句子
    sen_list = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], [2, 32, 1, 23, 1]]

    # 2- 对句子进行截断和填充
    max_length = 10
    """
        参数解释：
            sequences：要处理的句子列表
            maxlen：最大长度限制
            padding：填充的方式。默认pre，在句子前面填充；post在句子后面填充
            truncating：截断的方式。默认pre，在句子前面截断；post在句子后面截断
            value：用来进行填充的值。一般不会修改该参数，使用默认的0即可
    """
    # result = sequence.pad_sequences(sen_list, maxlen=max_length)
    result = sequence.pad_sequences(
        sen_list, maxlen=max_length, padding="post", truncating="post", value=6666
    )
    print(result)


def demo02():
    sen_list = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], [2, 32, 1, 23, 1]]

    # 1- 定义一个空列表用来存储规范化后的句子
    result_list = []

    max_length = 10

    # 2- 分别对每条句子进行处理
    for sen in sen_list:
        if len(sen) > max_length:
            # 超长的要截断
            result_list.append(sen[:max_length])
        else:
            # 短的要填充
            result_list.append(sen + [0] * (max_length - len(sen)))

    print(result_list)


if __name__ == "__main__":
    # demo01()

    demo02()
