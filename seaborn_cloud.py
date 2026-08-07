import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

import jieba.posseg as pseg
from wordcloud import WordCloud


def get_a_word(line):
    # 分词并且标注词性
    word_dict = pseg.lcut(line)
    # print(word_dict)

    result_line = []

    # 过滤形容词
    for word, pos in word_dict:
        if pos == "a":
            result_line.append(word)

    return result_line


def show_cloud(a_word_list):
    # 1- 创建词云对象
    wordcloud_obj = WordCloud(
        font_path="data/SimHei.ttf", max_words=100, background_color="white"
    )

    # 2- 词汇列表以空格分隔拼接成字符串
    word_str = " ".join(a_word_list)

    # 3- 绘制图形
    wordcloud_obj.generate(word_str)
    plt.figure(figsize=(10, 10))
    # bilinear：让文字边缘更加平滑
    plt.imshow(wordcloud_obj, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def word_cloud():
    # 1- 读取文件，并且取出评价内容
    df = pd.read_csv(os.path.join("data", "train.tsv"), encoding="utf-8", sep="\t")
    sentence_series = df["sentence"]

    # 2- 句子分词，过滤出形容词
    # 注意：map返回的是生成器，你要真的调用它，才会触发数据的产生
    a_word_list = list(chain(*map(get_a_word, sentence_series)))
    # a_word_list = list(chain(*map(lambda line: [word for word,pos in pseg.lcut(line) if pos=="a"], sentence_series)))
    print(a_word_list)

    # 3- 绘制词云
    show_cloud(a_word_list)


if __name__ == "__main__":
    word_cloud()
