import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from bert_model import BertClassifierModel
from config import Config
from data_preprocessing import build_dataloader

config = Config()


def eval_model(model):
    # 1- 加载验证集数据
    dev_dataloader = build_dataloader(datapath=config.dev_datapath, shuffle=False)

    # 2- 模型评估
    # 2.1- 定义用来计算准确率的变量
    all_pred_result = []  # 预测结果列表
    all_true_result = []  # 真实结果列表

    model.eval()
    with torch.no_grad():
        # 验证集使用单独的进度条：
        # 1. desc明确说明当前处于验证阶段；
        # 2. ncols限制宽度，避免进度条占满PyCharm控制台；
        # 3. leave=False表示验证结束后清除这一行，避免每验证一次就永久留下一个进度条。
        dev_progress = tqdm(
            dev_dataloader,
            desc="验证模型",
            unit="batch",
            ncols=120,
            leave=False,
        )

        for batch in dev_progress:
            # 2.2- 将数据发送到对应设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 2.3- 前向传播：预测
            pred_output = model(input_ids, attention_mask)
            pred_index = torch.argmax(pred_output, dim=-1)

            # cpu()：因为不涉及张量的计算，因此为了节约GPU资源，可以将数据转到CPU上再处理
            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    # 3- 计算评估指标
    f1score = f1_score(all_true_result, all_pred_result, average="macro")

    # 准确率
    accuracy = accuracy_score(all_true_result, all_pred_result)
    precision = precision_score(all_true_result, all_pred_result, average="macro")
    recall = recall_score(all_true_result, all_pred_result, average="macro")

    return f1score, accuracy, precision, recall


def train_and_eval():
    # 1- 加载数据
    train_dataloader = build_dataloader(datapath=config.train_datapath, shuffle=True)

    # 2- 创建模型
    model = BertClassifierModel().to(device=config.device)

    # 3- 损失函数对象
    loss = nn.CrossEntropyLoss()

    # 4- 优化器对象
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-5)

    # 5- 其他变量
    epochs = 1

    # 6- 训练
    model.train()  # 切换为训练模式
    best_f1score = 0.0  # f1score历史最高分

    for epoch in range(epochs):
        # 6.1- 训练指标
        total_loss = 0.0  # 总损失值

        # 每个Epoch创建一条训练进度条。
        # unit="batch"表示进度单位是批次；ncols限制控制台中的显示宽度。
        train_progress = tqdm(
            train_dataloader,
            desc=f"训练 Epoch {epoch + 1}/{epochs}",
            unit="batch",
            ncols=120,
        )

        for i, batch in enumerate(train_progress, start=1):
            # 6.2- 将数据发送到对应的设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 6.3- 前向传播：调用模型
            pred_output = model(input_ids, attention_mask)

            # 6.4- 算损失值
            loss_value = loss(pred_output, labels)

            # item()将单元素Loss张量转换成普通Python浮点数。
            # 修改原因：这里只需要累计并显示Loss数值，不应该让total_loss继续引用计算图。
            current_loss = loss_value.item()
            total_loss += current_loss

            # 6.5- 固定代码
            optimizer.zero_grad()
            # CrossEntropyLoss默认已经返回标量，直接backward()即可，不需要再调用sum()。
            loss_value.backward()
            optimizer.step()

            # 在同一条进度条右侧动态展示当前批次Loss和当前平均Loss。
            # set_postfix只更新进度条内容，不会像普通print一样把进度条切断。
            train_progress.set_postfix(
                loss=f"{current_loss:.4f}",
                avg_loss=f"{total_loss / i:.4f}",
            )

            # 6.6- 每隔100个批次对已训练的模型进行验证
            # i==len(train_dataloader)为了防止最后不够100个批次
            if i % 100 == 0 or i == len(train_dataloader):
                # 6.6.1- 调用评估函数
                f1score, accuracy, precision, recall = eval_model(model)

                # 进度条运行期间使用tqdm.write()，不要使用普通print()。
                # tqdm会先暂时移开训练进度条，完整打印验证结果后再恢复进度条，避免两者挤在同一行。
                tqdm.write(
                    f"[验证结果] "
                    f"批次={i}/{len(train_dataloader)} | "
                    f"F1={f1score:.4f} | "
                    f"准确率={accuracy:.4f} | "
                    f"精确率={precision:.4f} | "
                    f"召回率={recall:.4f}"
                )

                # 6.6.2- 如果验证后发现模型效果有提升（也就是f1score比上次的要大），那就保存模型
                if f1score > best_f1score:
                    torch.save(model.state_dict(), config.save_model)
                    best_f1score = f1score  # 更新历史最高分

                # 6.6.3- 将模型的模式切回为训练模式
                model.train()


if __name__ == "__main__":
    train_and_eval()
