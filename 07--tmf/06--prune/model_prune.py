# 模型剪枝：最核心代码
# 需求：对BERT全部Encoder中的Q、K、V权重进行30%的全局L1非结构化剪枝。

import sys
from pathlib import Path

import torch
from torch.nn.utils import prune

# 当前文件位于code_dl/07--tmf/06--prune，向上三级得到项目根目录code_dl。
# 显式加入sys.path后，无论PyCharm使用哪个工作目录，都能导入公共tools包。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    calculate_file_sha256,
    get_path_size_mb,
    print_model_compression_report,
    print_table,
    save_model_artifact,
    save_model_metadata,
)

from bert_model import BertClassifierModel
from config import Config
from data_preprocessing import build_dataloader
from model_eval import eval_model


def collect_prunable_parameters(model):
    """收集BERT所有Encoder层中Q、K、V三个线性层的weight。"""
    parameters_to_prune = []

    # 不把12层硬编码在代码里，直接读取当前BERT模型实际具有的Encoder层数。
    for encoder_layer in model.bert_model.encoder.layer:
        self_attention = encoder_layer.attention.self
        parameters_to_prune.extend(
            [
                (self_attention.query, "weight"),
                (self_attention.key, "weight"),
                (self_attention.value, "weight"),
            ]
        )

    return parameters_to_prune


def make_parameters_contiguous(parameters_to_prune):
    """兼容PyTorch剪枝内部对weight调用view(-1)的实现。"""
    converted_count = 0

    for module, parameter_name in parameters_to_prune:
        parameter = getattr(module, parameter_name)
        if parameter.is_contiguous():
            continue

        # 部分新版Transformers权重加载后可能是非连续内存。
        # prune.global_unstructured内部使用view(-1)，只接受连续张量；
        # contiguous()只调整内存布局，不改变权重数值、形状和预测结果。
        contiguous_parameter = torch.nn.Parameter(
            parameter.detach().contiguous(),
            requires_grad=parameter.requires_grad,
        )
        setattr(module, parameter_name, contiguous_parameter)
        converted_count += 1

    return converted_count


def calculate_sparsity(parameters_to_prune):
    """计算目标权重中零参数所占的比例。"""
    zero_count = 0
    parameter_count = 0

    for module, parameter_name in parameters_to_prune:
        parameter = getattr(module, parameter_name)
        zero_count += torch.count_nonzero(parameter == 0).item()
        parameter_count += parameter.numel()

    return zero_count / parameter_count if parameter_count else 0.0


def metrics_to_dict(f1, accuracy, precision, recall):
    """把评估函数返回值整理成公共对比表需要的字段。"""
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    print()
    config = Config()

    # 剪枝前后的评估必须使用同一份验证集，才能进行公平对比。
    dev_dataloader = build_dataloader(config.dev_datapath, shuffle=False)

    # 1- 加载剪枝前已经训练好的BERT分类模型
    if not config.before_prune_path.exists():
        raise FileNotFoundError(
            f"没有找到剪枝前模型：{config.before_prune_path}\n"
            "请先将03--bert/save_model/bert.pkl复制到该位置。"
        )

    # 先在CPU加载state_dict，再把完整模型移动到目标设备，兼容CPU和CUDA环境。
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

    parameters_to_prune = collect_prunable_parameters(model)
    converted_parameter_count = make_parameters_contiguous(parameters_to_prune)
    sparsity_before = calculate_sparsity(parameters_to_prune)

    before_f1, before_accuracy, before_precision, before_recall = eval_model(
        model, dev_dataloader
    )
    before_metrics = metrics_to_dict(
        before_f1, before_accuracy, before_precision, before_recall
    )

    # 模型结构信息和评估条件与权重分开保存。
    # 以后只读取metrics.json，就能知道这些指标是在哪份数据上计算出来的。
    model_config_metadata = {
        "model_class": type(model).__name__,
        # 只保存预训练模型目录名，避免把当前电脑的F盘绝对路径写入元数据。
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

    # before_prune_model.pkl已经存在，所以这里只补充同名metrics.json，
    # 不需要为了保存几个指标重新写入390MB的权重。
    save_model_metadata(
        config.before_prune_path,
        metrics=before_metrics,
        model_config=model_config_metadata,
        evaluation=evaluation_metadata,
        compression={"type": "none", "actual_sparsity": sparsity_before},
        extra={"source": "03--bert/save_model/bert.pkl"},
    )

    # 2- 对全部Encoder的Q、K、V权重进行全局非结构化剪枝
    """
    global_unstructured表示“全局”选择要删除的参数：
        1. 把12层Encoder中所有Q、K、V的weight放在一起比较；
        2. L1Unstructured根据权重绝对值判断重要程度；
        3. amount=0.3表示把整体中绝对值最小的30%权重置为0。

    这是非结构化剪枝：参数张量形状不变，只是部分数值变为0。
    因此普通dense state_dict文件大小通常不会明显缩小，普通CPU/GPU也未必加速。
    """
    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=config.prune_amount,
    )

    sparsity_after = calculate_sparsity(parameters_to_prune)

    # 3- 固化剪枝结果
    # prune.remove并不是恢复权重，而是把weight_orig * weight_mask的结果正式写回weight，
    # 同时删除临时的weight_orig和weight_mask重参数化字段，方便正常保存和加载state_dict。
    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)

    # 4- 使用相同验证集评估剪枝后的模型
    after_f1, after_accuracy, after_precision, after_recall = eval_model(
        model, dev_dataloader
    )
    after_metrics = metrics_to_dict(
        after_f1, after_accuracy, after_precision, after_recall
    )

    # 5- 保存剪枝后的state_dict和同名metrics.json
    # 权重文件仍是标准state_dict；评估指标、数据集版本和剪枝信息放在JSON中。
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
    )

    # 6- 打印稀疏率和剪枝范围
    print_table(
        title="BERT全局非结构化剪枝信息",
        columns=["项目", "剪枝前", "剪枝后"],
        rows=[
            ["目标剪枝比例", "0.00%", f"{config.prune_amount:.2%}"],
            ["Q/K/V权重实际稀疏率", f"{sparsity_before:.2%}", f"{sparsity_after:.2%}"],
            [
                "参与剪枝的Linear层数",
                len(parameters_to_prune),
                len(parameters_to_prune),
            ],
            [
                "转为连续内存的权重数",
                converted_parameter_count,
                converted_parameter_count,
            ],
        ],
    )

    # 7- 使用和模型量化、模型蒸馏相同的统一表格，对比剪枝前后的指标与文件大小。
    print_model_compression_report(
        original_metrics=before_metrics,
        compressed_metrics=after_metrics,
        original_size_mb=get_path_size_mb(config.before_prune_path),
        compressed_size_mb=get_path_size_mb(config.after_prune_path),
        original_name="剪枝前模型",
        compressed_name="剪枝后模型",
        title="BERT模型剪枝实验",
    )
