import sys
from pathlib import Path

import torch

# 当前文件位于code_dl/07--tmf/03--bert，向上三级可以得到项目根目录code_dl。
# 显式加入项目根目录后，无论从PyCharm还是终端运行，都可以稳定导入tools。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import get_path_size_mb, print_model_compression_report

from bert_model import BertClassifierModel
from bert_train_and_eval import eval_model
from config import Config


# 注意：动态量化（Dynamic Quantization，DQ）推荐在CPU环境下运行。
if __name__ == "__main__":
    # 1- 查看当前PyTorch支持的旧版Eager量化引擎
    """
    常见量化引擎：
        none：没有量化硬件加速。
        onednn：Intel提供的深度学习CPU计算库。
        x86：面向x86架构CPU的量化实现。
        fbgemm：Meta提供的x86 CPU量化计算库。

    当前Windows环境实测只提供onednn，因此下面明确使用onednn。
    这也是旧版动态量化在当前电脑上比TorchAO新API快很多的重要原因之一。
    """
    print(f"当前支持的量化引擎：{torch.backends.quantized.supported_engines}")
    torch.backends.quantized.engine = "onednn"

    # 2- 加载项目配置
    config = Config()

    # 3- 加载并评估量化前的FP32模型
    # 3.1- 创建与训练阶段完全相同的模型结构。
    model = BertClassifierModel()

    # 3.2- 加载训练好的state_dict。
    # map_location="cpu"：把模型权重加载到CPU，因为动态INT8量化主要用于CPU推理。
    # weights_only=True：当前文件只包含模型参数，不需要反序列化其他Python对象。
    model.load_state_dict(
        torch.load(config.save_model, map_location="cpu", weights_only=True)
    )

    # 3.3- 切换到评估模式，关闭Dropout等只在训练阶段使用的行为。
    model.eval()

    # 3.4- 使用验证集计算量化前指标。
    original_f1, original_accuracy, original_precision, original_recall = eval_model(
        model
    )
    original_metrics = {
        "accuracy": original_accuracy,
        "precision": original_precision,
        "recall": original_recall,
        "f1": original_f1,
    }

    # 4- 使用旧版PyTorch API进行动态INT8量化
    """
    torch.quantization.quantize_dynamic()参数说明：

    model=model：
        指定需要量化的原始FP32模型。

    qconfig_spec={torch.nn.Linear}：
        指定只量化模型中的nn.Linear线性层。

        BERT中可以被量化的Linear主要包括：
        1. 多头注意力里的Query、Key、Value三个投影层；
        2. 多头注意力结果的输出投影层；
        3. 前馈神经网络中的两个Linear层；
        4. BERT Pooler中的Linear层；
        5. 当前文本分类模型最后的Linear分类层。

        Embedding、LayerNorm、Dropout等不是Linear，所以不会被这个配置量化。

    dtype=torch.qint8：
        指定量化权重使用有符号8位量化整数。
        qint8不是普通业务计算中的torch.int8，而是PyTorch专门用于量化张量的类型；
        量化张量除了INT8数据，还需要保存scale、zero_point等量化参数。

    inplace=False：
        不直接修改原始model，而是返回一个新的量化模型。
        因此量化结束后可以同时保留：
            model                -> 原始FP32模型
            quantization_model   -> 动态INT8模型
        这样才能公平比较压缩前后的准确率和模型大小。

    动态量化中的“动态”主要是指：
        Linear权重提前保存为INT8；激活值在每次前向传播时，根据当前输入动态量化。
        因为不需要提前统计激活值范围，所以不需要额外准备校准数据集。
    """
    quantization_model = torch.quantization.quantize_dynamic(
        model=model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )

    # 5- TorchAO新版API对照（当前Windows CPU实测很慢，所以只保留为注释）
    """
    新版TorchAO核心写法如下。将来迁移时可以取消注释，但当前示例不执行它：

        from copy import deepcopy
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )

        torchao_model = deepcopy(model)
        quantize_(
            model=torchao_model,
            config=Int8DynamicActivationInt8WeightConfig(),
            device="cpu",
        )

    新旧API的主要区别：
        1. 旧quantize_dynamic()返回一个新的量化模型，当前代码更简单；
        2. 新quantize_()原地修改模型并返回None，因此需要先复制模型；
        3. 新API扩展量化方案更方便，但当前Windows CPU上的BERT实测约50秒/批次；
        4. 旧API虽然被标记弃用，但当前onednn执行路径速度明显更适合本示例。
    """

    # 6- 使用同一份验证集评估量化后的模型
    quantization_model.eval()
    quantized_f1, quantized_accuracy, quantized_precision, quantized_recall = (
        eval_model(quantization_model)
    )
    quantized_metrics = {
        "accuracy": quantized_accuracy,
        "precision": quantized_precision,
        "recall": quantized_recall,
        "f1": quantized_f1,
    }

    # 7- 打印模型结构
    # 可以观察原来的nn.Linear已经变成DynamicQuantizedLinear。
    print(quantization_model)

    # 8- 保存量化后的模型
    # 旧版动态量化模型可以直接保存整个模型对象，加载时更加直观。
    # 因为这是自己本地生成且可信的完整模型，加载时需要使用weights_only=False。
    quantized_model_path = Path("save_model/quantization_model.pkl")
    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantization_model, quantized_model_path)

    # 9- 打印压缩前后的统一对比表
    # 必须先保存INT8模型，才能读取两个模型文件在磁盘中的真实大小。
    # 两个模型使用的是同一份验证集和同一种指标算法，因此结果可以直接比较。
    print_model_compression_report(
        original_metrics=original_metrics,
        compressed_metrics=quantized_metrics,
        original_size_mb=get_path_size_mb(config.save_model),
        compressed_size_mb=get_path_size_mb(quantized_model_path),
        original_name="原始FP32模型",
        compressed_name="动态INT8模型",
    )
