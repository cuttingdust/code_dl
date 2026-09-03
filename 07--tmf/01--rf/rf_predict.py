import pandas as pd

# ml
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF类：将词变成词向量
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle  # 专门读写pickle类型的文件

from config import Config

# 基础配置信息
config = Config()

if __name__ == "__main__":
    # 1- 加载验证集数据
    df = pd.read_csv(config.process_dev_datapath, encoding="UTF-8")
    print(df.head())

    words = df["words"]
    label = df["label"]

    # 2- 加载训练好的模型
    with open(config.rf_model_path, mode="rb") as f:
        model: RandomForestClassifier = pickle.load(f)

    with open(config.tfidf_model_path, mode="rb") as f:
        tf_obj: TfidfVectorizer = pickle.load(f)

    # 3- 验证集数据特征提取
    x = tf_obj.transform(words)
    print(x)

    # 4- 模型预测
    y_pred = model.predict(x)
    print("精确率：", precision_score(label, y_pred, average="macro"))
    print("召回率：", recall_score(label, y_pred, average="macro"))
    print("F1值：", f1_score(label, y_pred, average="macro"))

    # 5- 预测结果输出到文件【可选】
    df["y_pred"] = y_pred
    df.to_csv(
        config.predict_result_path, mode="w", encoding="UTF-8", index=False, header=True
    )
