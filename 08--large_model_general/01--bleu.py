import math

from nltk.translate.bleu_score import sentence_bleu


# 计算 Bleu 指标
def cumulative_bleu(reference, candidate):
    """
    计算bleu的 1234，weights对应的权重
    """
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

    print("bleu 1-gram: %f" % bleu_1_gram)
    print("bleu 2-gram: %f" % bleu_2_gram)
    print("bleu 3-gram: %f" % bleu_3_gram)
    print("bleu 4-gram: %f" % bleu_4_gram)

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


print()
# 生成文本
generated_text = "it is a nice day today".split(" ")
print(type(generated_text))
print(generated_text)

# 参考文本列表
reference_texts = ["today is a nice day".split(" ")]
print(type(reference_texts))
print(reference_texts)

c_bleu = cumulative_bleu(reference_texts, generated_text)

# 打印结果
print("The Bleu score is:", c_bleu)
# The Bleu score is: (0.8333333333333334, 0.6, 0.5, 0.3333333333333333)
bleu = sentence_bleu(reference_texts, generated_text, weights=(0.25, 0.25, 0.25, 0.25))
print(f"接口返回结果：{bleu}")

print("手动计算结果：", math.exp(sum([0.25 * math.log(p) for p in c_bleu])))
