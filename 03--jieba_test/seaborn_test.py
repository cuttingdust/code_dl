import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1- 读取文件
    """
    注意：需要指定具体的分隔符sep，否则会报错pandas.errors.ParserError: Error tokenizing data. C error: Expected 6 fields in line 12, saw 8
    """
    train_df = pd.read_csv(
        os.path.join("../data", "train.tsv"), encoding="utf-8", sep="\t"
    )
    dev_df = pd.read_csv(os.path.join("../data", "dev.tsv"), encoding="utf-8", sep="\t")

    # 2- 创建一个画布，并在画布中创建两个子图
    """
    plt.subplots(1, 2)：
        1：画布中有1行子图
        2：画布中有2列子图

    返回值：
        fig：整个画布对象
        graphs：保存两个子图对象的数组
            graphs[0]：左边的第一个子图
            graphs[1]：右边的第二个子图

    figsize=(14, 6)：设置整个画布的宽度为14英寸、高度为6英寸。
    """
    plt.style.use("fivethirtyeight")
    fig, graphs = plt.subplots(1, 2, figsize=(14, 6))

    # 3- 在左边的子图中展示训练集标签数量
    # ax=graphs[0]：明确告诉Seaborn把图画到第一个子图中。
    sns.countplot(x="label", data=train_df, ax=graphs[0])
    graphs[0].set_title("train")
    graphs[0].set_xlabel("label")
    graphs[0].set_ylabel("count")

    # 4- 在右边的子图中展示验证集标签数量
    # ax=graphs[1]：明确告诉Seaborn把图画到第二个子图中。
    sns.countplot(x="label", data=dev_df, ax=graphs[1])
    graphs[1].set_title("dev")
    graphs[1].set_xlabel("label")
    graphs[1].set_ylabel("count")

    # 5- 自动调整两个子图之间的间距，避免标题和坐标文字重叠。
    fig.tight_layout()

    # 6- 最后只调用一次show()，两个子图会在同一个窗口中同时显示。
    plt.show()
