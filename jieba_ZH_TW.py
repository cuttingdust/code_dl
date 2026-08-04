import jieba

"""
    什么情况下需要用户自定义词典？
    答：当你真的是一名NLP开发工程师或者要开发公司内部的搜索引擎的时候才需要；根据公司业务场景进行针对性的分词才需要
"""


def demo01():
    content = "煩惱即是菩提，我暫且不提"
    result = jieba.lcut(content)
    print(result)

    content = "煩恼即是菩提，我暫且不提"
    result = jieba.lcut(content)
    print(result)


def demo02():
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 加载自定义词典
    jieba.load_userdict(r"./data/my_dict.txt")
    result = jieba.lcut(content)
    print(result)


if __name__ == "__main__":
    # 繁体字分词
    demo01()

    # 用户自定义词典
    demo02()
