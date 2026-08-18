import os

# 注意：1- 代码放在整个文件的最上面；2- 0需要是字符串的类型
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib

# from keras.src.legacy.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer

"""
    获得词向量的one-hot方式总结：
        优点：实现、理解简单
        缺点：
            1- 词向量是一个稀疏向量（大多数值为0，信息密度低），会浪费存储和计算资源
            2- 对多义词的处理不好。例如：过过过过过过，中的3对过的词向量是同一个
"""


def demo01():
    # 1- 准备语料数据
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
    # vocabs = ["过过", "过过", "过过"]   # 3个过过的one-hot是同一个

    # 2- 训练词汇映射器
    # 2.1- 创建词汇映射器
    my_tokenizer = Tokenizer(num_words=len(vocabs))
    # 2.2- 用语料库训练
    my_tokenizer.fit_on_texts(vocabs)
    # 2.3- 得到训练好的词和索引的映射关系
    # word_index类型是字典类型。key是词，value是词索引，索引从1开始
    word_dict = my_tokenizer.word_index
    print(type(word_dict))
    print(word_dict)

    # 3- 分别获得每个词的one-hot词向量
    for word in vocabs:
        # 3.1- 初始化全0的列表
        one_hot = [0] * len(word_dict)
        # 3.2- 获得词在词汇映射器中的索引位置
        index = word_dict[word] - 1
        # 3.3- 将指定位置的值设置为1即可
        one_hot[index] = 1

        print(f"{word}-->{one_hot}")

    # 4- 保存训练好的模型
    joblib.dump(my_tokenizer, r"../model/my_tokenizer.pkl")


def demo02():
    word = "李宗盛"

    # 1- 加载训练好的模型
    my_tokenizer = joblib.load(r"../model/my_tokenizer.pkl")

    # 2- 得到某个词的one-hot词向量
    # 2.1- 得到训练好的词和索引的映射关系
    word_dict = my_tokenizer.word_index
    # 2.2- 初始化全0的列表
    one_hot = [0] * len(word_dict)
    # 2.3- 获得词在词汇映射器中的索引位置
    index = word_dict[word] - 1
    # 2.4- 将指定位置的值设置为1即可
    one_hot[index] = 1

    print(f"{word}-->{one_hot}")


# 基于纯Python入门试学班的代码实现
def demo03():
    # 1- 准备语料数据
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]

    # 2- 获得每个词的词向量
    for word in vocabs:
        # 2.1- 初始化全0的列表
        one_hot = [0] * len(vocabs)
        # 2.2- 获得词对应的索引
        index = vocabs.index(word)
        # 2.3- 将对应位置的值设置为1
        one_hot[index] = 1
        print(f"{word}-->{one_hot}")


if __name__ == "__main__":
    # 1- 训练模型：也就是训练词汇映射器
    # print("-" * 50)
    # demo01()

    # 2- 使用训练好的模型
    # print("-" * 50)
    # demo02()

    # 3- 用最基本的方式实现
    print("-" * 50)
    demo03()
