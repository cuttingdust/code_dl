"""
预训练模型使用总结：
    1- pipeline管道：
        优点：代码开发非常简单
        缺点：底层高度封装，可调整的超参数少
        使用：一般用来快速验证预训练模型/大模型是否满足业务需求

    2- AutoModel自动模型：
        优点：代码相对比较简单，有一定可以可调整的超参数
        缺点：相对具体模型来说，可调整的超参数相对较少

    3- 指定模型：
        优点：可调整的超参数很多，能够针对具体的大模型进行指定参数的微调。每种大模型的可调整的参数都是不一样
        缺点：比较灵活，不同的大模型可调整的参数有区别
        使用：针对业务场景需要比较高的情况，推荐使用
        后续课程：LoRA、QLoRA
"""

import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM

if __name__ == "__main__":
    # 1- 创建模型的实例对象
    model_path = r"PretrainedModel/bert-base-chinese"
    # 创建词汇映射器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)

    # 2- 准备数据
    content = "我想明天去[MASK]家吃饭"

    # 3- 处理数据
    data_tensor = tokenizer(content, return_tensors="pt")

    # 4- 调用模型
    model.eval()
    result = model(**data_tensor)

    # 5- 结果解析
    # 5.1- 获得预测概率词最高的词索引
    pred_word_index = torch.argmax(result.logits[0][6]).item()
    # 5.2- 词索引转成词
    pred_word = tokenizer.convert_ids_to_tokens(pred_word_index)
    # 5.3- 输出
    print(f"填充词的索引：{pred_word_index}，对应内容：{pred_word}")
