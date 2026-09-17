"""剪枝后 BERT 新闻分类模型的加载与预测服务。

这个文件只负责两件事：

1. 从 ``save_model/after_prune_model.pkl`` 导入剪枝后的模型参数；
2. 把新闻标题转换成张量并执行预测。

命令行测试、Flask 后端和 Streamlit 前端都会复用这里的 ``predict``，
避免在多个文件中重复模型加载和推理代码。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import BertTokenizer

# 本文件位于 code_dl/07--tmf/06--prune。
# 显式加入项目根目录后，无论从 PyCharm 还是终端启动，都可以导入公共 tools。
APP_DIR = Path(__file__).resolve().parent
# APP_DIR 是 06--prune 目录：parents[0] 是 07--tmf，parents[1] 才是 code_dl。
PROJECT_ROOT = APP_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import get_model_metadata_path, load_model_metadata

from bert_model import BertClassifierModel
from config import Config

config = Config()


def load_pruned_model() -> BertClassifierModel:
    """导入剪枝后的模型并切换到预测模式。

    ``map_location`` 的作用是把权重加载到当前可用设备：

    - 有 CUDA 时加载到 GPU；
    - 没有 CUDA 时加载到 CPU；
    - 即使模型原来在 GPU 上保存，也可以在 CPU 电脑上读取。
    """
    if not config.after_prune_path.is_file():
        raise FileNotFoundError(
            "没有找到剪枝后的模型："
            f"{config.after_prune_path}\n"
            "请先运行 model_prune.py 或 model_prune_plus.py 生成模型。"
        )

    model = BertClassifierModel().to(config.device)
    state_dict = torch.load(
        config.after_prune_path,
        map_location=config.device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    # eval() 会关闭 Dropout 等只在训练阶段使用的行为。
    # 当前模型虽然没有自定义 Dropout，但 BERT 内部存在 Dropout，预测前必须调用。
    model.eval()
    return model


# 模块第一次被导入时加载一次模型和 Tokenizer。
# Flask 后续收到多个请求时会重复使用它们，不会每次请求都读取 390MB 模型文件。
model = load_pruned_model()
tokenizer = BertTokenizer.from_pretrained(config.bert_path)


def get_model_info() -> dict[str, Any]:
    """返回当前预测服务使用的模型信息和已有评估指标。"""
    model_info: dict[str, Any] = {
        "model_name": "BERT 剪枝模型",
        "model_path": str(config.after_prune_path),
        "device": str(config.device),
    }

    # 元数据文件用于展示剪枝实验的 Accuracy、Recall、F1 等结果。
    # 它不是模型预测的必要条件，所以不存在时仍然允许服务正常启动。
    metadata_path = get_model_metadata_path(config.after_prune_path)
    if metadata_path.is_file():
        metadata = load_model_metadata(config.after_prune_path)
        model_info["metrics"] = metadata.get("metrics", {})
        model_info["compression"] = metadata.get("compression", {})

    return model_info


def predict(news_data: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """预测一条新闻标题所属的类别，并返回概率最高的若干类别。

    :param news_data: 请求数据，例如 ``{"title": "新闻标题"}``。
    :param top_k: 返回概率最高的前几个类别，默认返回 Top 3。
    :return: 标题、最终类别、置信度和 Top K 分类概率。
    """
    if not isinstance(news_data, dict):
        raise ValueError("请求数据必须是字典，例如：{'title': '新闻标题'}")

    title = news_data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title 必须是非空字符串")
    title = title.strip()

    # return_tensors="pt" 让 Tokenizer 直接返回 PyTorch 张量，
    # 不需要再手动调用 torch.tensor(...) 转换。
    model_inputs = tokenizer(
        title,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    model_inputs = {
        name: tensor.to(config.device) for name, tensor in model_inputs.items()
    }

    with torch.inference_mode():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )

        # 模型输出 logits（原始分数），softmax 将其转换成总和为 1 的分类概率。
        probabilities = torch.softmax(logits, dim=-1)[0]
        top_k = max(1, min(int(top_k), config.classname_len))
        top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions = [
        {
            "class_name": config.classname_list[class_index],
            "probability": round(float(probability), 6),
        }
        for probability, class_index in zip(
            top_probabilities.cpu().tolist(),
            top_indices.cpu().tolist(),
        )
    ]

    return {
        "title": title,
        "pred_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["probability"],
        "top_predictions": top_predictions,
    }


if __name__ == "__main__":
    test_news = {"title": "体验2D巅峰 倚天屠龙记十大创新概览"}
    print(predict(test_news))
