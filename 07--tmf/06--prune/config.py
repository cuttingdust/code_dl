from pathlib import Path

import torch


class Config:
    def __init__(self):
        # 当前剪枝示例所在目录。后面的路径都基于当前文件计算，
        # 因此从PyCharm、终端或其他工作目录启动时都不会找错文件。
        self.base_dir = Path(__file__).resolve().parent

        # 1- 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2- 原始文件路径
        self.dev_datapath = self.base_dir / "data/dev.txt"
        self.class_datapath = self.base_dir / "data/class.txt"

        # 3- 数据加载器参数
        self.batch_size = 64  # 64条新闻
        self.max_length = 32  # 句子中词的个数最多是32个词

        # 4- Bert预训练模型的路径
        self.bert_path = (
            self.base_dir / "../../PretrainedModel/bert-base-chinese"
        ).resolve()

        # 5- 目标值文件解析
        with self.class_datapath.open(mode="r", encoding="UTF-8") as file:
            self.classname_list = [line.strip() for line in file]
        self.classname_len = len(self.classname_list)

        # 6- 模型保存目录
        self.save_model_dir = self.base_dir / "save_model"
        self.save_model_dir.mkdir(parents=True, exist_ok=True)

        # 7- 剪枝前模型路径
        # 该文件复制自03--bert/save_model/bert.pkl，二者使用完全相同的网络结构。
        self.before_prune_path = self.save_model_dir / "before_prune_model.pkl"

        # 8- 剪枝后模型的保存路径
        self.after_prune_path = self.save_model_dir / "after_prune_model.pkl"

        # 9- 剪枝比例：删除所有目标权重中绝对值最小的30%。
        self.prune_amount = 0.3
