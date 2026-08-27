import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch

from transformers import AutoTokenizer  # 分词器
from transformers import AutoModelForSequenceClassification  # 序列分类
from transformers import AutoModel  # 通用模型加载类


def text_classification():
    # 1- 创建类的实例对象
    model_path = r"PretrainedModel/chinese_sentiment"
    # 创建分词器对象
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 创建预训练模型对象
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 2- 准备数据
    content = "我爱黑马"

    # 3- 分词：将文本内容转成模型需要的数据类型，也就是张量
    """
    参数解释：
        text：要进行分词处理的内容
        return_tensors：返回结果的数据类型。pt 张量；np ndarray数组。推荐pt
        truncation：是否要对超过max_length长度的句子进行截断
        max_length：能够处理的最长句子的长度
        padding：
            如果是布尔值，同时句子的长度没有超过max_length，不会进行短句子的填充
            如果是max_length，同时句子的长度没有超过max_length，会进行短句子的填充，使用词索引为0的进行填充

            举例：
                参数值为True 结果 [[ 101, 2769, 4263, 7946, 7716,  102]]
                参数值为max_length 结果 [[ 101, 2769, 4263, 7946, 7716,  102,    0,    0,    0,    0]]
    """
    data_tensor = tokenizer.encode(
        text=content,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=10,
    )

    print("-" * 40)
    print(type(data_tensor))
    print(data_tensor.shape)
    print(data_tensor)
    print("-" * 40)

    # 4- 调用
    # 4.1- 切换为预测模式
    model.eval()

    # 4.2- 调用模型
    pred_result = model(data_tensor)
    print(type(pred_result))
    """
        SequenceClassifierOutput(loss=None, logits=tensor([[-1.5430, -0.9715, -0.4824,  0.1655,  1.0699]],
       grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    """
    print(pred_result)

    # 4.3- 获得预测概率最高的那个类别对应的索引
    print(pred_result[0].argmax(dim=-1))


def text_feature_extraction():
    # 1- 创建模型实例对象
    model_path = r"PretrainedModel\bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # 2- 准备数据
    sens = ["你是谁", "人生该如何起头"]

    # 3- 将数据处理成张量
    # Transformers 5.x中直接调用tokenizer对象即可。
    # text是第一句话，text_pair是第二句话；分词器会自动按照下面的结构拼接：
    # [CLS] 第一句话 [SEP] 第二句话 [SEP] [PAD] ...
    data_tensor = tokenizer(
        text=sens[0],
        text_pair=sens[1],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30,
    )

    """
        返回值解释：
            1- input_ids：句子对应的词索引
            2- token_type_ids：词索引来源于的句子索引。句子索引从0开始
            3- attention_mask：注意力掩码。0表示不看input_ids对应位置的词索引；1反之
            {
      'input_ids': tensor([[ 101,  872, 3221, 6443,  102,  782, 4495, 6421, 1963,  862, 6629, 1928,
              102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                0,    0,    0,    0,    0,    0]]),

    'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0]])}
    """
    print(data_tensor)

    # 4- 调用模型提取文本特征
    # eval()将模型切换到预测模式，关闭Dropout等只在训练时使用的行为。
    model.eval()

    # 当前只做预测，不需要反向传播；no_grad()可以减少内存占用。
    with torch.no_grad():
        # data_tensor是一个类似字典的BatchEncoding对象。
        # **表示把其中的数据按照参数名展开后传给模型，相当于：
        # model(
        #     input_ids=data_tensor["input_ids"],
        #     token_type_ids=data_tensor["token_type_ids"],
        #     attention_mask=data_tensor["attention_mask"],
        # )
        output = model(**data_tensor)

    # last_hidden_state保存每个token经过BERT编码后的特征向量。
    # 当前形状应为[1, 30, 768]：1组句子对，30个token位置，每个位置768维。
    print(f"模型输出类型：{type(output)}")
    print(f"文本特征形状：{output.last_hidden_state.shape}")
    print(output.last_hidden_state)


if __name__ == "__main__":
    print("")

    # 1- 文本分类
    # text_classification()

    # 2- 文本特征提取
    text_feature_extraction()
