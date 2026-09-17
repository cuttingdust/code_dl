"""模型权重与评估元数据的统一保存、读取工具。

推荐把模型拆成两个文件：

1. ``模型名.pkl``：只保存PyTorch ``state_dict``，保持标准加载方式。
2. ``模型名.metrics.json``：保存指标、数据集、模型配置和压缩信息。

这样既能在不加载大模型的情况下直接查看评估结果，也能在指标更新时避免
重新写入几百MB的模型权重文件。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

MODEL_ARTIFACT_SCHEMA_VERSION = 1


def get_model_metadata_path(model_path: str | Path) -> Path:
    """根据权重路径生成对应的元数据路径。

    示例：``after_prune_model.pkl`` -> ``after_prune_model.metrics.json``。
    """
    model_path = Path(model_path)
    return model_path.with_suffix(".metrics.json")


def calculate_file_sha256(file_path: str | Path) -> str:
    """分块计算文件SHA-256，用来标识本次评估使用的数据集版本。"""
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def _json_default(value: Any) -> Any:
    """把常见配置对象转换成JSON能够保存的数据。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("元数据中的Tensor必须只有一个元素")
        return value.item()

    raise TypeError(f"元数据包含无法转换成JSON的类型：{type(value).__name__}")


def _normalize_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """把NumPy浮点数等数值统一转换成普通Python float。"""
    normalized_metrics = {}
    for metric_name, metric_value in metrics.items():
        normalized_metrics[str(metric_name)] = float(metric_value)
    return normalized_metrics


def build_model_metadata(
    model_path: str | Path,
    *,
    metrics: Mapping[str, float],
    model_config: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    compression: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一份可移植的模型评估元数据字典。"""
    model_path = Path(model_path)

    metadata = {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        # 只记录文件名，不记录当前电脑的绝对路径，复制目录后仍然有效。
        "model_file": model_path.name,
        "model_size_bytes": model_path.stat().st_size if model_path.exists() else None,
        "metrics": _normalize_metrics(metrics),
        "model_config": dict(model_config or {}),
        "evaluation": dict(evaluation or {}),
        "compression": dict(compression or {}),
        "extra": dict(extra or {}),
    }
    return metadata


def save_model_metadata(
    model_path: str | Path,
    *,
    metrics: Mapping[str, float],
    model_config: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    compression: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """为已经存在的模型权重保存同名评估元数据JSON。"""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重文件不存在：{model_path}")

    metadata_path = get_model_metadata_path(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_model_metadata(
        model_path,
        metrics=metrics,
        model_config=model_config,
        evaluation=evaluation,
        compression=compression,
        extra=extra,
    )

    # 先写临时文件，再原子替换正式JSON，防止程序中断后留下半个JSON文件。
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return metadata_path


def save_model_artifact(
    model: nn.Module,
    model_path: str | Path,
    *,
    metrics: Mapping[str, float],
    model_config: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    compression: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """保存模型state_dict，并为它生成同名评估元数据JSON。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存CPU权重，之后可以通过map_location加载到任意可用设备。
    cpu_state_dict = {
        name: parameter.detach().cpu() for name, parameter in model.state_dict().items()
    }
    torch.save(cpu_state_dict, model_path)

    metadata_path = save_model_metadata(
        model_path,
        metrics=metrics,
        model_config=model_config,
        evaluation=evaluation,
        compression=compression,
        extra=extra,
    )
    return model_path, metadata_path


def load_model_metadata(model_path: str | Path) -> dict[str, Any]:
    """读取模型对应的评估元数据，不需要加载大型权重文件。"""
    metadata_path = get_model_metadata_path(model_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"模型评估元数据不存在：{metadata_path}")

    with metadata_path.open(mode="r", encoding="utf-8") as file:
        metadata = json.load(file)

    schema_version = metadata.get("schema_version")
    if schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "不支持的模型元数据版本："
            f"{schema_version}，当前支持{MODEL_ARTIFACT_SCHEMA_VERSION}"
        )
    return metadata


def load_model_artifact(
    model: nn.Module,
    model_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """加载state_dict到现有模型，并返回该模型保存的评估元数据。"""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重文件不存在：{model_path}")

    state_dict = torch.load(
        model_path,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=strict)
    return load_model_metadata(model_path)


__all__ = [
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "build_model_metadata",
    "calculate_file_sha256",
    "get_model_metadata_path",
    "load_model_artifact",
    "load_model_metadata",
    "save_model_artifact",
    "save_model_metadata",
]
