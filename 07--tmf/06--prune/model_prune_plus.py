# BERT模型剪枝增强观察版
#
# 与model_prune.py的区别：
#   1. model_prune.py着重演示完整、可靠的剪枝和指标对比流程；
#   2. 当前文件在相同流程基础上，额外展示权重样本、被剪位置和每层稀疏率；
#   3. 方便从数值层面观察“剪枝究竟把哪些参数变成了0”。

import sys
from pathlib import Path

import torch
from torch.nn.utils import prune

# 当前文件位于code_dl/07--tmf/06--prune，向上三级得到项目根目录code_dl。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    calculate_file_sha256,
    get_path_size_mb,
    print_model_compression_report,
    print_panel,
    print_table,
    save_model_artifact,
    save_model_metadata,
)

from bert_model import BertClassifierModel
from config import Config
from data_preprocessing import build_dataloader
from model_eval import eval_model
from model_prune import (
    calculate_sparsity,
    collect_prunable_parameters,
    make_parameters_contiguous,
    metrics_to_dict,
)


def get_weight_sample(weight, rows=5, cols=5):
    """复制权重左上角的样本，防止后续剪枝原地修改观察数据。"""
    return weight[:rows, :cols].detach().cpu().clone()


def print_weight_sample(weight_sample, title):
    """使用公共Rich面板显示权重样本。"""
    print_panel(
        title=title,
        content=str(weight_sample),
        style="cyan",
    )


def calculate_layer_sparsities(model):
    """分别计算12个Encoder层中Q、K、V整体的稀疏率。

    参考代码只统计了每层Query的零权重，但真正的剪枝目标包含Query、Key和Value。
    这里把每一层的三个权重矩阵放在一起统计，范围与实际剪枝操作保持一致。
    """
    layer_sparsities = []

    for layer_index, encoder_layer in enumerate(model.bert_model.encoder.layer):
        self_attention = encoder_layer.attention.self
        layer_weights = [
            self_attention.query.weight,
            self_attention.key.weight,
            self_attention.value.weight,
        ]

        total_count = sum(weight.numel() for weight in layer_weights)
        zero_count = sum(
            torch.count_nonzero(weight == 0).item() for weight in layer_weights
        )

        layer_sparsities.append(
            {
                "layer": layer_index,
                "total_count": total_count,
                "zero_count": zero_count,
                "sparsity": zero_count / total_count if total_count else 0.0,
            }
        )

    return layer_sparsities


def count_target_parameters(parameters_to_prune):
    """统计本次参与全局剪枝的Q/K/V权重总数。"""
    return sum(
        getattr(module, parameter_name).numel()
        for module, parameter_name in parameters_to_prune
    )


if __name__ == "__main__":
    print()
    config = Config()

    # 1- 准备验证数据和剪枝前模型
    # 剪枝前后必须使用同一份DataLoader，评估结果才可以直接比较。
    dev_dataloader = build_dataloader(config.dev_datapath, shuffle=False)

    if not config.before_prune_path.exists():
        raise FileNotFoundError(
            f"没有找到剪枝前模型：{config.before_prune_path}\n"
            "请先将03--bert/save_model/bert.pkl复制到该位置。"
        )

    model = BertClassifierModel()
    model.load_state_dict(
        torch.load(
            config.before_prune_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    model = model.to(config.device)
    model.eval()

    # 2- 收集全部12层Encoder中的Q、K、V权重
    parameters_to_prune = collect_prunable_parameters(model)

    # 新版Transformers加载的部分权重不是连续内存，而PyTorch剪枝内部使用view(-1)。
    # 转为连续内存只改变存储排列，不改变参数数值和预测结果。
    converted_parameter_count = make_parameters_contiguous(parameters_to_prune)

    target_parameter_count = count_target_parameters(parameters_to_prune)
    sparsity_before = calculate_sparsity(parameters_to_prune)
    layer_sparsities_before = calculate_layer_sparsities(model)

    # 只抽样展示第一层Query左上角5×5权重。
    # clone()非常重要，否则剪枝后before样本也可能跟着原权重一起变化。
    first_query_weight = model.bert_model.encoder.layer[0].attention.self.query.weight
    query_sample_before = get_weight_sample(first_query_weight)

    # 3- 评估剪枝前模型
    before_f1, before_accuracy, before_precision, before_recall = eval_model(
        model, dev_dataloader
    )
    before_metrics = metrics_to_dict(
        before_f1, before_accuracy, before_precision, before_recall
    )

    model_config_metadata = {
        "model_class": type(model).__name__,
        "pretrained_model": config.bert_path.name,
        "max_length": config.max_length,
        "class_names": config.classname_list,
        "class_count": config.classname_len,
    }
    evaluation_metadata = {
        "dataset": str(config.dev_datapath.relative_to(config.base_dir)),
        "dataset_sha256": calculate_file_sha256(config.dev_datapath),
        "split": "dev",
        "sample_count": len(dev_dataloader.dataset),
        "metric_average": "macro",
    }

    save_model_metadata(
        config.before_prune_path,
        metrics=before_metrics,
        model_config=model_config_metadata,
        evaluation=evaluation_metadata,
        compression={"type": "none", "actual_sparsity": sparsity_before},
        extra={"source": "03--bert/save_model/bert.pkl"},
    )

    # 4- 执行30%全局L1非结构化剪枝
    # “全局”表示把全部36个Q/K/V权重矩阵放在一起比较绝对值，
    # 最终把整体中绝对值最小的30%置为0，而不是要求每一层都刚好剪掉30%。
    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=config.prune_amount,
    )

    sparsity_after = calculate_sparsity(parameters_to_prune)
    layer_sparsities_after = calculate_layer_sparsities(model)

    # 5- 固化剪枝结果
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)

    # 获取剪枝后的同一块权重，以及样本中“原来非0、剪枝后变0”的位置。
    first_query_weight = model.bert_model.encoder.layer[0].attention.self.query.weight
    query_sample_after = get_weight_sample(first_query_weight)
    sample_pruned_mask = (query_sample_before != 0) & (query_sample_after == 0)
    sample_pruned_count = torch.count_nonzero(sample_pruned_mask).item()

    # 6- 评估剪枝后模型
    after_f1, after_accuracy, after_precision, after_recall = eval_model(
        model, dev_dataloader
    )
    after_metrics = metrics_to_dict(
        after_f1, after_accuracy, after_precision, after_recall
    )

    # 增强观察版与基础版使用相同的剪枝算法，所以保存到相同的剪枝后模型路径。
    save_model_artifact(
        model,
        config.after_prune_path,
        metrics=after_metrics,
        model_config=model_config_metadata,
        evaluation=evaluation_metadata,
        compression={
            "type": "global_l1_unstructured",
            "target": "BERT Encoder self-attention Q/K/V weights",
            "requested_prune_amount": config.prune_amount,
            "actual_sparsity": sparsity_after,
            "pruned_linear_layer_count": len(parameters_to_prune),
        },
        extra={
            "experiment_script": "model_prune_plus.py",
            "sample_shape": list(query_sample_before.shape),
            "sample_pruned_count": sample_pruned_count,
        },
    )

    # 7- 展示第一层Query的权重变化
    print_weight_sample(
        query_sample_before,
        "剪枝前：第1层Query权重左上角5×5",
    )
    print_weight_sample(
        query_sample_after,
        "剪枝后：第1层Query权重左上角5×5",
    )
    print_weight_sample(
        sample_pruned_mask.to(torch.int8),
        "样本剪枝位置（1表示该位置被剪成0）",
    )

    # 全局30%剪枝不代表每层正好30%，因此逐层打印实际稀疏率。
    layer_rows = []
    for before, after in zip(layer_sparsities_before, layer_sparsities_after):
        layer_rows.append(
            [
                before["layer"],
                before["total_count"],
                f"{before['sparsity']:.2%}",
                f"{after['sparsity']:.2%}",
                after["zero_count"],
            ]
        )

    print_table(
        title="各Encoder层Q/K/V权重稀疏率",
        columns=["Encoder层", "参数总数", "剪枝前", "剪枝后", "剪枝后零参数数"],
        rows=layer_rows,
    )

    print_table(
        title="BERT全局非结构化剪枝增强信息",
        columns=["项目", "结果"],
        rows=[
            ["参与剪枝的Linear层数", len(parameters_to_prune)],
            ["参与剪枝的权重总数", target_parameter_count],
            ["转为连续内存的权重数", converted_parameter_count],
            ["目标剪枝比例", f"{config.prune_amount:.2%}"],
            ["剪枝前整体稀疏率", f"{sparsity_before:.2%}"],
            ["剪枝后整体稀疏率", f"{sparsity_after:.2%}"],
            ["5×5样本中被剪权重数", sample_pruned_count],
        ],
    )

    # 8- 使用与量化、蒸馏和基础剪枝版相同的指标对比表
    print_model_compression_report(
        original_metrics=before_metrics,
        compressed_metrics=after_metrics,
        original_size_mb=get_path_size_mb(config.before_prune_path),
        compressed_size_mb=get_path_size_mb(config.after_prune_path),
        original_name="剪枝前模型",
        compressed_name="剪枝后模型",
        title="BERT模型剪枝增强观察实验",
    )
