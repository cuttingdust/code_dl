import pandas as pd

# ml
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF类：将词变成词向量
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle  # 专门读写pickle类型的文件

from config import Config

# 基础配置信息
config = Config()

if __name__ == "__main__":
    # 1- 读取预处理好的数据
    df = pd.read_csv(config.predict_result_path)

    # 注意：为了加快演示速度
    # df = df.sample(20000)
    print(df.head())

    # 2- 得到特征数据和目标值
    words = df["words"]
    label = df["label"]

    # 3- 特征工程：用TF-IDF来表示词向量
    # 3.1- 加载停用词：去除掉新闻标题中无意义的内容
    stop_words_list = open(config.stopwords_datapath, encoding="UTF-8").read().split()
    print(stop_words_list)
    # 3.2- 创建TF-IDF算法模型实例对象
    tf_obj = TfidfVectorizer(stop_words=stop_words_list)
    # 3.3- 对文本内容进行向量化处理
    x = tf_obj.fit_transform(words)
    print(x.shape)
    print("处理后的特征名称：", tf_obj.get_feature_names_out())

    # vocabulary_是TF-IDF学习得到的“词 -> 特征列索引”字典：
    #   1- 前面的内容是词，例如“体验”“2d”“巅峰”。
    #   2- 后面的数字是这个词在TF-IDF特征矩阵中的列索引，不是词频，也不是TF-IDF权重。
    #      例如 {'体验': 3813} 表示“体验”这个词对应x矩阵的第3813列。
    #   3- 某篇文本中“体验”的TF-IDF值，需要查看x[文本行索引, 3813]。
    #   4- 词和列索引也可以反向验证：
    #      tf_obj.get_feature_names_out()[3813] 的结果就是“体验”。
    #   5- 索引只是模型为词分配的列位置，数字大不代表词更重要，数字小也不代表出现次数更多。
    print("处理后的词汇表：", tf_obj.vocabulary_)
    print("处理后的词汇表大小：", len(tf_obj.vocabulary_))

    """
       TF-IDF处理后的数据解释：
           Coords	Values
           (0, 818)	0.4145423492151699

           1- 0：代表该词所在的句子索引
           2- 818：代表该词在TF-IDF特征矩阵中的列索引
           3- 0.4145423492151699：TF-IDF的值
    """
    print("处理后的词向量：", x)

    # 4- 数据划分：得到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, label, test_size=0.2, random_state=315, shuffle=True
    )

    # 5- 模型训练
    """
        参数解释：
            n_estimators：弱学习器的个数
            n_jobs：指定多少个线程数
    """
    model = RandomForestClassifier(n_estimators=100, n_jobs=10)
    model.fit(x_train, y_train)

    # 6- 模型评估
    y_pred = model.predict(x_test)
    """
        注意：我们现在是多分类问题，因此average的值不能是binary。
            macro：宏平均。适合多分类的场景，如果样本分布均衡推荐使用
            micro：微观平均。适合多分类的场景，如果样本分布不均衡推荐使用
    """
    print("精确率：", precision_score(y_test, y_pred, average="macro"))
    print("召回率：", recall_score(y_test, y_pred, average="macro"))
    print("F1值：", f1_score(y_test, y_pred, average="macro"))

    # 7- 保存训练好的模型
    with open(config.rf_model_path, mode="wb") as f:
        pickle.dump(model, f)

    with open(config.tfidf_model_path, mode="wb") as f:
        pickle.dump(tf_obj, f)
