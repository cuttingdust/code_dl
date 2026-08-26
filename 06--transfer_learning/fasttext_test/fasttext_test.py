# pip install fasttext-wheel
import fasttext

from pathlib import Path


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


def demo09():
    # 训练模型
    # ova：多标签多分类 问题简化成 单标签多分类的问题
    model = fasttext.train_supervised(
        input="data/cooking.pre.train", epoch=20, lr=0.1, wordNgrams=2, loss="ova"
    )

    model_path = Path("model")
    model_path.mkdir(parents=True, exist_ok=True)

    # 保存训练好的模型
    model.save_model("model/fasttext.pkl")

    # 加载训练好的模型
    model = fasttext.load_model("model/fasttext.pkl")
    pred_labels = model.predict(
        text=["how to seperate peanut oil from roasted peanuts at home?"], k=5
    )
    print(pred_labels)


if __name__ == "__main__":
    print("")
    # 1- 模型训练、预测、评估
    # print("-" * 40)
    # demo01()  # (3000, 0.13733333333333334, 0.05939166786795445)
    #
    # # 2- 数据基本处理
    # print("-" * 40)
    # demo02()  # (3000, 0.167, 0.07222142136370188)
    #
    # # 3- 增加训练轮次
    # print("-" * 40)
    # demo03()  # (3000, 0.48966666666666664, 0.21176300994666283)
    #
    # # 4- 调整学习率
    # print("-" * 40)
    # demo04()  # (3000, 0.5896666666666667, 0.25500937004468793)
    #
    # # 5- 设置N-gram参数
    # print("-" * 40)
    # demo05()  # (3000, 0.5986666666666667, 0.25890154245351016)
    #
    # # 6- 调整损失函数：主要是提升了运行速度
    # print("-" * 40)
    # demo06()
    #
    # # 7- 自动超参数调优
    # print("-" * 40)
    # demo07()
    #
    # # 8- 多标签多分类问题
    # print("-" * 40)
    # demo08()

    # 9- 保存模型和重新加载模型
    print("-" * 40)
    demo09()

    print("#" * 50)
