import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 当前文件位于code_dl/07--tmf/05--distill，向上三级可以得到项目根目录code_dl。
# 显式加入项目根目录后，无论从PyCharm还是终端运行，都可以稳定导入tools。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    create_batch_progress,
    get_path_size_mb,
    print_evaluation_result,
    print_model_compression_report,
    update_progress_metrics,
)

from config import Config
from data_preprocessing import build_dataloader
from teacher_bert_model import BertTeacherModel
from student_bilstm_model import BiLSTMStudentModel

config = Config()


def eval_model(model, use_token_type_ids=False):
    # 1- 加载验证集数据
    dev_dataloader = build_dataloader(datapath=config.dev_datapath, shuffle=False)

    # 2- 模型评估
    # 2.1- 定义用来计算准确率的变量
    all_pred_result = []  # 预测结果列表
    all_true_result = []  # 真实结果列表

    model.eval()
    with torch.no_grad():
        # 单独显示验证进度；leave=False表示验证完成后清除这一行，
        # 避免每验证一次就在PyCharm控制台永久留下一个进度条。
        dev_progress = create_batch_progress(
            dev_dataloader,
            description="验证模型",
            leave=False,
        )

        for batch in dev_progress:
            # 2.2- 将数据发送到对应设备
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            labels = labels.to(device=config.device)

            # 2.3- 前向传播：预测
            # 教师BERT需要token_type_ids；学生BiLSTM只使用词索引和掩码。
            if use_token_type_ids:
                token_type_ids = token_type_ids.to(device=config.device)
                pred_output = model(input_ids, attention_mask, token_type_ids)
            else:
                pred_output = model(input_ids, attention_mask)
            pred_index = torch.argmax(pred_output, dim=-1)
            # cpu()：因为不涉及张量的计算，因此为了节约GPU资源，可以将数据转到CPU上再处理
            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    # 3- 计算评估指标
    f1score = f1_score(
        all_true_result,
        all_pred_result,
        average="macro",
    )

    # 准确率
    accuracy = accuracy_score(all_true_result, all_pred_result)
    precision = precision_score(
        all_true_result, all_pred_result, average="macro", zero_division=0
    )
    recall = recall_score(
        all_true_result, all_pred_result, average="macro", zero_division=0
    )

    return f1score, accuracy, precision, recall


def train_teacher_model():
    """重新训练教师BERT的分类层，并返回验证集表现最好的教师模型。"""
    train_dataloader = build_dataloader(
        datapath=config.train_datapath,
        shuffle=True,
    )

    teacher_model = BertTeacherModel().to(device=config.device)
    loss_fn = nn.CrossEntropyLoss()

    # BERT主体已经冻结，只把需要梯度的Linear分类层参数交给优化器。
    optimizer = torch.optim.AdamW(
        params=(p for p in teacher_model.parameters() if p.requires_grad),
        lr=5e-5,
    )

    epochs = 1
    best_f1score = -1.0
    validation_interval = 500

    for epoch in range(epochs):
        teacher_model.train()
        # BERT主体被冻结，让它保持eval模式可关闭Dropout，保证句子特征稳定。
        teacher_model.bert_model.eval()
        total_loss = 0.0
        train_progress = create_batch_progress(
            train_dataloader,
            description=f"教师训练 Epoch {epoch + 1}/{epochs}",
        )

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(
            train_progress, start=1
        ):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            labels = labels.to(config.device)

            pred_output = teacher_model(input_ids, attention_mask, token_type_ids)
            loss_value = loss_fn(pred_output, labels)
            current_loss = loss_value.item()
            total_loss += current_loss

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            update_progress_metrics(
                train_progress,
                loss=current_loss,
                avg_loss=total_loss / i,
            )

            if i % validation_interval == 0 or i == len(train_dataloader):
                f1score, accuracy, precision, recall = eval_model(
                    teacher_model, use_token_type_ids=True
                )
                print_evaluation_result(
                    batch_index=i,
                    total_batches=len(train_dataloader),
                    f1=f1score,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                )

                if f1score > best_f1score:
                    best_f1score = f1score
                    torch.save(teacher_model.state_dict(), config.teacher_save_model)

                teacher_model.train()
                teacher_model.bert_model.eval()

    # 后续蒸馏和最终报告都使用验证集F1最高的教师权重。
    teacher_model.load_state_dict(
        torch.load(
            config.teacher_save_model,
            map_location=config.device,
            weights_only=True,
        )
    )
    teacher_model.eval()

    teacher_f1, teacher_accuracy, teacher_precision, teacher_recall = eval_model(
        teacher_model, use_token_type_ids=True
    )
    teacher_metrics = {
        "accuracy": teacher_accuracy,
        "precision": teacher_precision,
        "recall": teacher_recall,
        "f1": teacher_f1,
    }
    return teacher_model, teacher_metrics


def train_and_eval():
    # 第一次训练时保存目录可能不存在，提前创建，避免torch.save()报错。
    Path(config.teacher_save_model).parent.mkdir(parents=True, exist_ok=True)
    Path(config.student_save_model).parent.mkdir(parents=True, exist_ok=True)

    # 1- 先重新训练教师分类模型，并记录正确的教师基准指标。
    teacher_model, teacher_metrics = train_teacher_model()

    # 2- 加载学生模型蒸馏训练数据
    dataloader = build_dataloader(
        datapath=config.train_datapath,
        shuffle=True,
    )

    # 3- 新建学生模型。每次实验都重新初始化，不使用旧学生权重。
    student_model = BiLSTMStudentModel().to(device=config.device)

    # 3- 损失函数对象：用来计算硬标签的损失值
    loss = nn.CrossEntropyLoss()

    # 4- 优化器对象
    optim = torch.optim.AdamW(params=student_model.parameters(), lr=5e-5)

    # 5- 其他变量
    epochs = 1
    best_f1score = 0.0  # f1值历史最高分
    validation_interval = 500

    T = 2  # 软标签中的温度超参数
    alpha = 0.7  # 软硬标签的平衡权重系数

    # 6- 训练
    # 6.1- 模型模式设置
    teacher_model.eval()  # 注意：教师模型已经训练好，不允许进行反向传播
    student_model.train()  # 注意：学生模型需要进行反向传播，更新w和b，为了学会教师模型的能力

    for epoch in range(epochs):
        # 当前Epoch所有批次的蒸馏总损失，用来计算进度条中的平均损失。
        total_loss = 0.0

        # 每个Epoch只保留一条训练进度条，右侧动态更新损失值，
        # 避免使用普通print()后在控制台产生大量零散输出。
        train_progress = create_batch_progress(
            dataloader,
            description=f"蒸馏训练 Epoch {epoch + 1}/{epochs}",
        )

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(
            train_progress, start=1
        ):
            # 6.2- 将训练数据发送到设备
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            token_type_ids = token_type_ids.to(device=config.device)
            labels = labels.to(device=config.device)

            # 6.3- 教师模型_前向传播
            with torch.no_grad():
                teacher_pred = teacher_model(input_ids, attention_mask, token_type_ids)

            # 6.4- 学生模型_前向传播
            student_pred = student_model(input_ids, attention_mask)

            # 6.5- 计算KL散度值（软标签损失）
            """
            知识蒸馏使用的KL散度公式：

                KL(P_teacher || Q_student)
                    = Σ P_teacher(i) * [log P_teacher(i) - log Q_student(i)]

            符号说明：
                P_teacher：教师模型给出的普通概率分布
                Q_student：学生模型给出的普通概率分布
                i：某一个类别

            从公式可以看到：
                1- 教师模型需要提供普通概率P_teacher，因此使用softmax；
                2- 学生模型需要提供log(Q_student)，因此使用log_softmax。

            PyTorch的functional.kl_div默认要求：
                input  = 对数概率log(Q)，这里传学生模型的q；
                target = 普通概率P，这里传教师模型的p；
                log_target=False，表示target不是对数概率。

            因此下面计算的是：
                KL(教师概率分布 || 学生概率分布)

            如果教师模型也使用log_softmax得到对数概率，那么必须把
            log_target设置为True，不能只替换softmax而保持其他参数不变。

            温度T的作用：
                logits除以T后，T越大，softmax概率分布越平缓，学生可以看到
                教师对非正确类别的相对判断，也就是比硬标签更丰富的软信息。
            """
            q = torch.log_softmax(student_pred / T, dim=-1)
            p = torch.softmax(teacher_pred / T, dim=-1)

            # 计算KL散度值，也就是软标签损失值。
            """
            参数解释：
                input：学生模型的对数概率log(Q_student)
                target：教师模型的普通概率P_teacher
                reduction="batchmean"：先对所有类别求和，再按照批次求平均，
                                       与KL散度的数学定义对应
                log_target=False：target不是对数概率；当前p由softmax得到，
                                  所以这里必须设置为False
            """
            kl_loss = torch.nn.functional.kl_div(
                input=q,
                target=p,
                reduction="batchmean",
                log_target=False,
            )

            # 6.6- 硬标签损失值
            # 注意：是学生模型的预测概率，与样本的目标值算损失
            hard_loss = loss(student_pred, labels)

            # 6.7- 蒸馏的总损失值
            # 蒸馏总损失公式：
            # L = (1-α) * HardLoss + α * T² * KL(P_teacher || Q_student)
            # T²用于补偿温度缩放造成的梯度变小，使软标签损失在不同T下仍有合适的影响力。
            distill_loss = (1 - alpha) * hard_loss + alpha * T**2 * kl_loss

            # item()只提取普通浮点数用于显示，不会让损失累计过程继续引用计算图。
            current_loss = distill_loss.item()
            total_loss += current_loss

            # 6.8- 固定代码
            optim.zero_grad()
            # 反向传播 -> 同时优化hard_loss和kl_loss -> 同时优化硬标签和软标签 -> 教师模型同时从硬标签和软标签层面教会学生模型对结果的预测能力
            distill_loss.sum().backward()
            optim.step()

            # 在同一条进度条右侧动态显示当前损失和当前平均损失。
            update_progress_metrics(
                train_progress,
                loss=current_loss,
                avg_loss=total_loss / i,
            )

            # 6.9- 每隔100个批次或最后一个批次，对学生模型进行验证
            if i % validation_interval == 0 or i == len(dataloader):
                # 6.9.1- 调用评估函数
                f1score, accuracy, precision, recall = eval_model(student_model)
                # 公共工具内部使用tqdm.write()，不会破坏正在运行的训练进度条。
                print_evaluation_result(
                    batch_index=i,
                    total_batches=len(dataloader),
                    f1=f1score,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                )

                # 6.9.2- 如果验证后发现模型效果有提升（也就是f1score比上次的要大），那就保存模型
                if f1score > best_f1score:
                    best_f1score = f1score
                    torch.save(student_model.state_dict(), config.student_save_model)

                # 6.9.3- 将模型的模式切回为训练模式
                student_model.train()

    # 7- 完整训练结束后，加载验证集F1最高的学生模型进行最终对比。
    # 这样表格比较的是“最佳学生模型”，而不是最后一个批次的临时参数。
    student_model.load_state_dict(
        torch.load(
            config.student_save_model,
            map_location=config.device,
            weights_only=True,
        )
    )

    student_f1, student_accuracy, student_precision, student_recall = eval_model(
        student_model
    )
    student_metrics = {
        "accuracy": student_accuracy,
        "precision": student_precision,
        "recall": student_recall,
        "f1": student_f1,
    }

    # 与model_quantize_dynamic.py使用同一个公共打印工具，
    # 统一展示蒸馏前后模型效果和模型文件大小的变化。
    print_model_compression_report(
        original_metrics=teacher_metrics,
        compressed_metrics=student_metrics,
        original_size_mb=get_path_size_mb(config.teacher_save_model),
        compressed_size_mb=get_path_size_mb(config.student_save_model),
        original_name="教师BERT模型",
        compressed_name="学生BiLSTM模型",
        title="BERT模型蒸馏实验",
    )


if __name__ == "__main__":
    print()
    train_and_eval()
