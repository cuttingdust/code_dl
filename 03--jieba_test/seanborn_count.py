import os
import jieba
import seaborn as sns
import pandas as pd
from itertools import chain

if __name__ == "__main__":
    df = pd.read_csv(os.path.join("../data", "train.tsv"), encoding="utf-8", sep="\t")
    # print("-" * 50)
    # print(df.head())

    # map_result = map(lambda line: jieba.lcut(line), df["sentence"])
    # print("-" * 50)
    # print(map_result)
    # print(list(map_result))
    # print(*map_result)

    # word_set = set(chain(*map_result))
    # print("-" * 50)
    # print(word_set)
    # print(list(word_set))

    # 合并的写法版本
    word_set = set(chain(*map(lambda line: jieba.lcut(line), df["sentence"])))

    print(
        f"词汇总个数: {len(word_set)}",
    )
