class Config:
    def __init__(self):
        # 1- 原始文件路径
        # 训练集
        self.train_datapath = "../00--data/data/train.txt"
        # 测试集
        self.test_datapath = "../00--data/data/test.txt"
        # 验证集
        self.dev_datapath = "../00--data/data/dev.txt"
        # 类别种类
        self.class_datapath = "../00--data/data/class.txt"
        # 训练集
        self.stopwords_datapath = "../00--data/data/stopwords.txt"

        # 2- 分词预处理后文件存放路径
        # 训练集
        self.process_train_datapath = "data/process_train_data.csv"
        # 测试集
        self.process_test_datapath = "data/process_test_data.csv"
        # 验证集
        self.process_dev_datapath = "data/process_dev_data.csv"

        # ------------------------ 理解：简化代码 ------------------------
        # 1- 原始文件路径
        self.original_datapath_dict = {
            "train": self.train_datapath,
            "test": self.test_datapath,
            "dev": self.dev_datapath,
        }
        # 2- 分词预处理后文件存放路径
        self.process_datapath_dict = {
            "train": self.process_train_datapath,
            "test": self.process_test_datapath,
            "dev": self.process_dev_datapath,
        }

        # 3- 训练好的模型保存路径
        self.tfidf_model_path = "save_model/tfidf.pkl"
        self.rf_model_path = "save_model/rf.pkl"

        # 4- 预测结果存储路径
        self.predict_result_path = "result/dev.csv"
