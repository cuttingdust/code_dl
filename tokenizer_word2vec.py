"""
如果不管是安装fasttext还是fasttext-wheel都失败，那么原因是python版本过高。
操作步骤如下：
    1- 先进入对应的虚拟环境
    2- 安装低版本的python解释器
        conda install python=3.10
    3- 安装fasttext
        pip install fasttext-wheel -i https://mirrors.aliyun.com/pypi/simple/
"""

import fasttext


def demo01():
    # 1- 使用无监督学习训练模型
    """
    为什么这里只能使用无监督学习？
    答：因为数据中没有明确标记目标值。有监督学习对文件内容有严格要求，有__label__
    """
    # model = fasttext.train_unsupervised("data/fil9")
    model = fasttext.train_unsupervised("data/gz03ag")

    # 2- 保存训练好的模型
    # model.save_model("model/word2vec.pkl")
    model.save_model("model/word2vec2.pkl")


def demo02():
    # 1- 加载训练好的模型
    # model = fasttext.load_model("model/word2vec.pkl")
    # model = fasttext.load_model("model/word2vec2.pkl")
    model = fasttext.load_model("model/word2vec3.pkl")

    # 2- 获得某个词的词向量
    word_vec = model.get_word_vector("hello")
    print(word_vec)


def demo03():
    # 1- 加载训练好的模型
    # model = fasttext.load_model("model/word2vec.pkl")
    # model = fasttext.load_model("model/word2vec2.pkl")
    model = fasttext.load_model("model/word2vec3.pkl")

    # 2- 获得相近的几个词
    # 会从词形、词义方面进行查找。
    # result_list = model.get_nearestsheepdog_neighbors("dog")
    result_list = model.get_nearest_neighbors("cat")

    # 3- 输出
    print(result_list)


def demo04():
    """
    参数解释：
        input：训练集数据路径
        model：具体的模式。默认是skipgram。还可以设置cbow
        lr：初始的学习率
        dim：词向量的维度。该值越大训练越耗时，但是存储的信息越丰富
        epoch：训练的轮次
        thread：并发训练的线程个数
    """
    model = fasttext.train_unsupervised(
        input="data/gz03ag", model="cbow", lr=0.1, dim=100, epoch=1, thread=10
    )

    # 保存训练好的模型
    model.save_model("model/word2vec3.pkl")


if __name__ == "__main__":
    # 1- 训练并保存模型
    # demo01()

    # 2- 获得词的词向量
    # demo02()

    # 3- 获得相近的词
    demo03()

    # 4- 超参数调优设置【掌握】
    # demo04()
