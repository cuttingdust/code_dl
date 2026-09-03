# 数据预处理：将一条句子分词得到一个个的词

import pandas
import jieba
import pandas as pd

from config import Config

# 1- 加载配置文件
config = Config()


def preprocessing(datapath):
    print(f"datapath --> {datapath}")

    # 1- 读取文件内容
    df = pd.read_csv(datapath, encoding="utf-8", sep="\t", names=["text", "label"])
    print(df.head())

    # 2- 对新闻标题进行分词，然后进行句子长度的处理只保留前30个词
    # line代表的是每行中的text内容，也就是新闻标题
    df["words"] = df["text"].apply(lambda line: " ".join(jieba.lcut(line)[:30]))

    # 3- 保存处理后的文件
    # 3.1- 训练集
    if "train" in datapath:
        df.to_csv(
            config.process_train_datapath, sep=",", header=True, index=False, mode="w"
        )
        print("train数据预处理完成，保存成功")

    # 3.2- 测试集
    if "test" in datapath:
        df.to_csv(
            config.process_test_datapath,
            sep=",",
            header=True,
            index=False,
            mode="w",
            encoding="utf-8",
        )
        print("test数据预处理完成，保存成功")

    # 3.3- 验证集
    if "dev" in datapath:
        df.to_csv(
            config.process_dev_datapath,
            sep=",",
            header=True,
            index=False,
            mode="w",
            encoding="utf-8",
        )
        print("dev数据预处理完成，保存成功")


def preprocessing_plus(task_type, datapath):
    print(f"datapath --> {datapath}, task_type --> {task_type}")

    # 1- 读取对应的文件
    df = pd.read_csv(datapath, sep="\t", encoding="utf-8", names=["text", "label"])
    print(df.head())

    # 2- 对text句子进行分词，并且只取前30个词
    df["words"] = df["text"].apply(lambda line: " ".join(jieba.lcut(line)[:30]))

    # 3- 将处理后的DF写入文件进行存储
    if task_type in datapath:
        df.to_csv(
            config.process_datapath_dict.get(task_type),
            sep=",",
            header=True,
            index=False,
            mode="w",
            encoding="UTF-8",
        )
        print(f"{task_type}数据预处理完成，保存成功")


if __name__ == "__main__":
    # 2- 获得训练集、测试集、验证集的数据路径；调用数据预处理函数
    # ------------------ 普通版写法 ------------------
    # # 2.1- 训练集的处理
    # datapath = config.train_datapath
    # preprocessing(datapath)
    #
    # # 2.2- 测试集的处理
    # datapath = config.test_datapath
    # preprocessing(datapath)
    #
    # # # 2.3- 验证集的处理
    # datapath = config.dev_datapath
    # preprocessing(datapath)

    # ------------------ 优化版写法 ------------------
    for task_type, datapath in config.original_datapath_dict.items():
        preprocessing_plus(task_type, datapath)
