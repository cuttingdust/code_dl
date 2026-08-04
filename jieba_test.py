import jieba
from pygments.lexer import words
from redis.commands.search import result

"""
    推荐要求大家重点掌握jieba.lcut()方法即可
"""
content = (
    "传智hello教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
)


# 【推荐】精确模式
def demo01():
    # Ctrl+Q：看方法解释
    words = jieba.lcut(content)
    print(words)

    # 了解：不带l的cut返回结果是generator生成器
    result = jieba.cut(content)
    print(result)
    print(type(result))
    print(list(result))


# 全词模式
def demo02():
    # cut_all默认是False。如果设置为True，那就是全词模式，分词更加细致
    words = jieba.lcut(content, cut_all=True)
    print(words)


def demo03():
    # 注意：没有cut_all参数
    words = jieba.lcut_for_search(content)
    print(words)


if __name__ == "__main__":
    # 【推荐】精确模式
    print("-" * 50)
    demo01()

    # 全词模式
    print("-" * 50)
    demo02()

    # 搜索引擎模式
    print("-" * 50)
    demo03()
