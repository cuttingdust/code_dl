class Config:
    def __init__(self):

        # 1- 原始文件路径
        # 训练集
        self.train_datapath = "../00--data/data/train.txt"
        # 测试集
        self.test_datapath = "../00--data/data/test.txt"
        # 验证集
        self.dev_datapath = "../00--data/data/dev.txt"
        # 目标值类别
        self.class_datapath = "../00--data/data/class.txt"

        # 2- 预处理后的文件路径：字符级
        # 训练集
        self.process_char_train_datapath = "data/process_char_train.txt"
        # 测试集
        self.process_char_test_datapath = "data/process_char_test.txt"
        # 验证集
        self.process_char_dev_datapath = "data/process_char_dev.txt"

        # 3- 预处理后的文件路径：词级
        # 训练集
        self.process_word_train_datapath = "data/process_word_train.txt"
        # 测试集
        self.process_word_test_datapath = "data/process_word_test.txt"
        # 验证集
        self.process_word_dev_datapath = "data/process_word_dev.txt"

        # 4- 将class.txt文件内容转成字典。格式如下：
        # {0: 'finance', 1: 'realty'...}
        self.id2label = {
            index: class_name.strip()
            for index, class_name in enumerate(
                open(self.class_datapath, mode="r", encoding="UTF-8")
            )
        }

        # 5- 训练好模型的保存路径
        self.model_char_manual_train = "save_model/char_manual_train.pkl"
        self.model_char_auto_train = "save_model/char_auto_train.pkl"
        self.model_word_manual_train = "save_model/word_manual_train.pkl"
        self.model_word_auto_train = "save_model/word_auto_train.pkl"


if __name__ == "__main__":
    config = Config()
    print(config.id2label)
