import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from transformers import pipeline  # 管道形式
from transformers.pipelines import SUPPORTED_TASKS

print("-" * 50)
print(SUPPORTED_TASKS.keys())
print("-" * 50)


def text_classification():
    # 1- 加载预训练模型
    """
    参数解释：
        task：业务的任务类型。这里是文本分类，可以传递text-classification或sentiment-analysis，推荐使用text-classification
            注意：其他任务的task值去pipeline源代码中找
        model：预训练模型的路径。推荐前面加上r，避免转义
    """
    model = pipeline(
        task="text-classification", model="PretrainedModel/chinese_sentiment"
    )  # 5分类问题

    # model = pipeline(
    #     task="text-classification", model=r"PretrainedModel/bert-base-chinese"
    # )  # 二分类。好评是1，差评是0

    # 2- 预测
    print("-" * 40)
    pred_result = model("这家餐馆的卫生还行，就是有点油")
    print(type(pred_result))
    print(pred_result)

    print("-" * 40)
    pred_result = model("这家餐馆的卫生太差了，吃了拉稀，非常不推荐")
    print(pred_result)

    print("-" * 40)
    pred_result = model("我爱北京天安门，天安门上太阳升。")
    print(pred_result)

    print("-" * 40)
    pred_result = model("这家餐馆的伙食太好吃了，卫生也非常干净，十分好评")
    print(pred_result)


def text_feature_extraction():
    # 1- 加载预训练模型
    model = pipeline(
        task="feature-extraction", model="PretrainedModel/bert-base-chinese"
    )

    # 2- 文本特征提取：先分词->以列表形式返回词向量
    # 返回的形状是 [1, 17, 768]。1是句子条数，17是因为对每个字进行分词加上句子的开始和结束；768词向量的维度数
    result = model("这家餐馆的卫生还行，就是有点油")

    print(type(result))
    print(result)

    print(torch.tensor(result).shape)


def fill_blank():
    # 1- 加载预训练模型#
    # model = pipeline(task="fill-mask", model=r"PretrainedModel\bert-base-chinese")
    model = pipeline(task="fill-mask", model=r"PretrainedModel\chinese-bert-wwm")

    # 2- 填空
    """
        注意：要进行填充的地方，必须写 [MASK]
    """
    content = "我想明天去[MASK]家吃饭。"
    result = model(content)
    print(type(result))
    print(result)


def text_q_and_a():
    """使用pipeline完成中文纯文本阅读理解。"""

    # 1- 创建“文本问答”管道。
    # 这个模型经过了中文机器阅读理解（MRC）任务微调，
    # 能够根据question，从context原文中抽取答案。
    model = pipeline(
        task="question-answering",
        model=r"PretrainedModel/chinese_pretrain_mrc_roberta_wwm_ext_large",
        device=-1,  # -1表示使用CPU；使用第一张显卡时可以改成0。
    )

    # 2- 准备原文和问题。
    context = "我叫张三，我是一个程序员，我的喜好是打篮球。"
    questions = ["我是谁？", "我是做什么的？", "我的爱好是什么？"]

    print(f"原文：{context}")
    print("-" * 50)

    # 3- 逐个提问。
    for question in questions:
        # 返回结果是一个字典，主要包含：
        # answer：从原文中抽取的答案；
        # score：模型对答案的置信度；
        # start和end：答案在原文中的字符起止位置。
        answer = model(
            question=question,
            context=context,
            # 中文不像英文那样用空格分词，关闭英文单词边界对齐更合适。
            align_to_words=False,
        )

        print(f"问题：{question}")
        print(f"答案：{answer['answer']}")
        print(f"置信度：{answer['score']:.4f}")
        print(f"字符位置：[{answer['start']}:{answer['end']}]")
        print("-" * 50)


def text_generation_q_and_a():
    """使用Transformers 5.x的text-generation管道完成中文生成式问答。"""

    # 1- 创建文本生成管道。
    # 注意：text-generation必须使用Qwen这类“生成式模型”，
    # 不能继续使用上面BERT结构的中文MRC问答模型。
    model = pipeline(
        task="text-generation",
        model=r"PretrainedModel/Qwen2.5-0.5B-Instruct",
        # device=-1,  # 当前CUDA/cuDNN存在兼容问题，先固定使用CPU验证管道。
        # dtype=torch.float32,
    )

    # 2- 为了和传统抽取式问答对比，继续使用完全相同的原文和问题。
    context = "我叫张三，我是一个程序员，我的喜好是打篮球。"
    questions = ["我是谁？", "我是做什么的？", "我的爱好是什么？"]

    print(f"原文：{context}")
    print("-" * 50)

    # 3- 逐个问题调用生成式模型。
    for question in questions:
        # text-generation不会接收单独的question和context参数，
        # 而是把任务要求、原文和问题一起组织成聊天消息。
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个中文阅读理解助手。"
                    "请严格根据用户提供的原文回答，只输出简短答案，不要解释。"
                    "如果原文没有答案，就回答：原文未提及。"
                ),
            },
            {
                "role": "user",
                "content": f"原文：{context}\n问题：{question}",
            },
        ]

        result = model(
            messages,
            do_sample=False,  # 关闭随机采样，让相同输入的结果更稳定。
            max_new_tokens=32,  # 限制回答长度，避免模型继续生成无关内容。
        )

        # 聊天模式下，generated_text保存完整消息列表；
        # 最后一条消息就是模型新生成的assistant回答。
        answer = result[0]["generated_text"][-1]["content"].strip()

        print(f"问题：{question}")
        print(f"答案：{answer}")
        print("-" * 50)


# 需要transformers4框架
def summary():
    # 1- 加载模型
    model = pipeline(task="summarization", model=r"PretrainedModel\distilbart-cnn-12-6")

    # 2- 提供语料库
    text = (
        "BERT is a transformers model pretrained on a large corpus of English data "
        "in a self-supervised fashion. This means it was pretrained on the raw texts "
        "only, with no humans labelling them in any way (which is why it can use lots "
        "of publicly available data) with an automatic process to generate inputs and "
        "labels from those texts. More precisely, it was pretrained with two objectives:Masked "
        "language modeling (MLM): taking a sentence, the model randomly masks 15% of the "
        "words in the input then run the entire masked sentence through the model and has "
        "to predict the masked words. This is different from traditional recurrent neural "
        "networks (RNNs) that usually see the words one after the other, or from autoregressive "
        "models like GPT which internally mask the future tokens. It allows the model to learn "
        "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models"
        " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to "
        "sentences that were next to each other in the original text, sometimes not. The model then "
        "has to predict if the two sentences were following each other or not."
    )

    summary_result = model(text)
    print(type(summary_result))
    print(summary_result)


def ner():
    # 1- 加载模型
    model = pipeline(
        task="ner",
        model=r"PretrainedModel\roberta-base-finetuned-cluener2020-chinese",
    )

    # 2- 提取实体
    print(model("鲁迅原名周树人，代表作有朝花夕拾，在商务部上班，今天他去故宫游览"))
    """
        B表示命名实体的开始，I是命名实体的中间内容

        {'entity': 'B-name', 'score': 0.9945884, 'index': 1, 'word': '鲁', 'start': 0, 'end': 1}, 
        {'entity': 'I-name', 'score': 0.99043053, 'index': 2, 'word': '迅', 'start': 1, 'end': 2}, 

        {'entity': 'B-name', 'score': 0.9791542, 'index': 5, 'word': '周', 'start': 4, 'end': 5}, 
        {'entity': 'I-name', 'score': 0.97904646, 'index': 6, 'word': '树', 'start': 5, 'end': 6}, 
        {'entity': 'I-name', 'score': 0.9797911, 'index': 7, 'word': '人', 'start': 6, 'end': 7}, 

        {'entity': 'I-organization', 'score': 0.3546072, 'index': 20, 'word': '务', 'start': 19, 'end': 20}, 
        {'entity': 'I-organization', 'score': 0.32793036, 'index': 21, 'word': '部', 'start': 20, 'end': 21}, 

        {'entity': 'B-scene', 'score': 0.881768, 'index': 29, 'word': '故', 'start': 28, 'end': 29}, 
        {'entity': 'I-scene', 'score': 0.91957027, 'index': 30, 'word': '宫', 'start': 29, 'end': 30}
    """


if __name__ == "__main__":
    print("")
    # 1- 文本分类
    # text_classification()

    # 2- 文本特征提取
    # text_feature_extraction()

    # 3- 完型填空
    # fill_blank()

    # 4- 阅读理解
    # 4-1 传统抽取式文本问答（仅Transformers 4.x支持该pipeline任务）
    # text_q_and_a()

    # 4-2 Transformers 5.x生成式文本问答
    # text_generation_q_and_a()

    # 5- 文本摘要
    # summary()

    # 6- NER命名实体识别
    ner()
