import torch
from transformers import BertTokenizer

from bert_model import BertClassifierModel
from config import Config

# 1- 加载训练好的模型
config = Config()

model = BertClassifierModel().to(config.device)
# 注意：不要用变量接受返回结果
model.load_state_dict(torch.load(config.save_model))
tokenizer = BertTokenizer.from_pretrained(config.bert_path)


# 2- 预测函数
def predict(news_data):
    # 数据预处理
    title = news_data["title"]
    # title只有1条新闻，但tokenizer需要按批次处理，所以使用列表将它包装成一个批次。
    data_tensor = tokenizer(
        [title],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
    )

    input_ids = torch.tensor(data_tensor.input_ids).to(config.device)
    attention_mask = torch.tensor(data_tensor.attention_mask).to(config.device)

    # 模型预测
    with torch.no_grad():
        # 前向传播：预测
        pred_output = model(input_ids=input_ids, attention_mask=attention_mask)
        # 获得概率最高的分类索引
        pred_index = torch.argmax(pred_output, dim=-1).item()
        # 索引转成类别名称
        pred_class_name = config.classname_list[pred_index]

        # 返回结果
    news_data["pred_class"] = pred_class_name

    return news_data


if __name__ == "__main__":
    print(predict({"title": "体验2D巅峰 倚天屠龙记十大创新概览"}))
