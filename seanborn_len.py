import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def demo01():
    # 1- 读取数据
    df = pd.read_csv(os.path.join("data", "train.tsv"), encoding="utf-8", sep="\t")
    # 2- 统计句子长度
    # 2.1- 获得句子列
    sentence_series = df["sentence"]
    # 2.2- 计算句子长度
    # 方式一：list(map)
    # df["length"] = list(map(lambda line: len(line), sentence_series))
    # print("-" * 50)
    # print(df.head())

    # 方式二：apply
    df["length"] = sentence_series.apply(lambda line: len(line))
    print("-" * 50)
    print(df.head())

    # 3- 绘制图形
    # 3.1- 绘制分布直方图
    plt.style.use("seaborn-v0_8")
    sns.set_palette("muted")
    sns.set_context("talk")
    sns.countplot(x="label", hue="length", data=df)
    plt.xticks([])
    plt.title("length_distribution")
    plt.show()

    # 3.2- 绘制趋势曲线


def demo02():
    # 1- 读取数据
    df = pd.read_csv(os.path.join("data", "train.tsv"), encoding="utf-8", sep="\t")

    # 2- 统计句子长度
    # 2.1- 获得句子列
    sentence_series = df["sentence"]
    # 2.2- 计算句子长度
    df["length"] = sentence_series.apply(lambda line: len(line))
    # 3- 绘制图形
    sns.stripplot(x="label", y="length", data=df, hue="label")
    plt.title("label_length")
    plt.show()


if __name__ == "__main__":
    # 句子长度分布
    # demo01()

    # 好评差评的句子长度分布
    demo02()
