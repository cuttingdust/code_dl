import torch


class Config:
    def __init__(self):
        # 1- 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2- 原始文件路径
        self.train_datapath = "data/train.txt"
        self.dev_datapath = "data/dev.txt"
        self.test_datapath = "data/test.txt"
        self.class_datapath = "data/class.txt"

        # 3- 数据加载器参数
        self.batch_size = 64  # 64条新闻
        self.max_length = 32  # 句子中词的个数最多是32个词

        # 4- Bert预训练模型的路径
        self.bert_path = r"..\..\PretrainedModel\bert-base-chinese"

        # 5- 目标值文件解析
        self.classname_list = [
            line.strip()
            for line in open(self.class_datapath, mode="r", encoding="UTF-8")
        ]
        self.classname_len = len(self.classname_list)

        # 6- 本次蒸馏实验重新训练并保存的【教师模型】路径
        self.teacher_save_model = r"save_model\bert.pkl"

        # 7- 训练好的【学生模型】的保存路径
        self.student_save_model = r"save_model\student_bert.pkl"


if __name__ == "__main__":
    config = Config()
    print(config.classname_list)
    print(config.classname_len)
