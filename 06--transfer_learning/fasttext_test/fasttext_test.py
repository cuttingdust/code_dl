# pip install fasttext-wheel
import fasttext

def demo01():
    # 1- 模型训练
    model = fasttext.train_supervised(input=r"data/cooking_train.txt")

    # 2- 模型预测
    # 参数解释：k指的是返回Top-K个预测概率最高的目标值。
    pred_labels = model.predict(
        text=["Which baking dish is best to bake a banana bread ?"],
        k=3,
    )
    print(type(pred_labels))  # 类型是嵌套元组
    print(pred_labels)

    pred_labels = model.predict(
        text=["how to seperate peanut oil from roasted peanuts at home?"],
        k=5,
    )
    print(pred_labels)

    # 3- 模型评估
    result = model.test(path="data/cooking_valid.txt")
    print(result)


# 将字母全部统一成小写，而且在标点符号前增加空格。
def demo02():
    # 训练模型
    model = fasttext.train_supervised(input="data/cooking.pre.train")

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


# 增加训练轮次。
def demo03():
    # 训练模型
    model = fasttext.train_supervised(
        input="data/cooking.pre.train",
        epoch=20,
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


# 调整学习率。
def demo04():
    # 训练模型
    model = fasttext.train_supervised(
        input="data/cooking.pre.train",
        epoch=20,
        lr=1,
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


# 设置N-gram参数。
def demo05():
    # 训练模型
    model = fasttext.train_supervised(
        input="data/cooking.pre.train",
        epoch=20,
        lr=1,
        wordNgrams=2,
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


def demo06():
    # 训练模型
    # hs：层次softmax
    model = fasttext.train_supervised(
        input="data/cooking.pre.train", epoch=20, lr=1, wordNgrams=2, loss="hs"
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


def demo07():
    # 训练模型
    # hs：层次softmax
    model = fasttext.train_supervised(
        input="data/cooking.pre.train",
        autotuneDuration=60 * 2,
        autotuneValidationFile="data/cooking.pre.valid",
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)


# 将 多标签多分类 问题简化成 单标签多分类的问题。每种分类单独进行训练
def demo08():
    # 训练模型
    # ova：多标签多分类 问题简化成 单标签多分类的问题
    model = fasttext.train_supervised(
        input="data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss="ova"
    )

    # 模型评估
    result = model.test(path="data/cooking.pre.valid")
    print(result)
