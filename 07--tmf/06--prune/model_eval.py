import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import Config
from data_preprocessing import build_dataloader

config = Config()


def eval_model(model, dev_dataloader=None):
    """使用验证集评估模型，返回顺序与03--bert中的评估函数保持一致。"""
    # 1- 加载验证集数据
    # 允许调用方传入已经创建好的DataLoader，剪枝前后便能复用完全相同的数据。
    if dev_dataloader is None:
        dev_dataloader = build_dataloader(
            datapath=config.dev_datapath,
            shuffle=False,
        )

    # 2- 模型评估
    # 2.1- 定义用来计算准确率的变量
    all_pred_result = []  # 预测结果列表
    all_true_result = []  # 真实结果列表

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="验证模型", ncols=100):
            # 2.2- 将数据发送到对应设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            labels = labels.to(device=config.device)

            # 2.3- 前向传播：预测
            pred_output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_index = torch.argmax(pred_output, dim=-1)
            # cpu()：因为不涉及张量的计算，因此为了节约GPU资源，可以将数据转到CPU上再处理
            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    # 3- 计算评估指标
    f1score = f1_score(
        all_true_result, all_pred_result, average="macro", zero_division=0
    )
    # 准确率
    accuracy = accuracy_score(all_true_result, all_pred_result)
    precision = precision_score(
        all_true_result, all_pred_result, average="macro", zero_division=0
    )
    recall = recall_score(
        all_true_result, all_pred_result, average="macro", zero_division=0
    )

    # 统一返回顺序：F1、Accuracy、Precision、Recall。
    return f1score, accuracy, precision, recall
