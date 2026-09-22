import math

# 定义语料库
sentences = [
    ["I", "have", "a", "pen"],
    ["He", "has", "a", "book"],
    ["She", "has", "a", "cat"],
]

# 定义语言模型
unigram = {
    "I": 1 / 100,
    "have": 1 / 100,
    "a": 3 / 100,
    "pen": 5 / 100,
    "He": 7 / 100,
    "has": 21 / 100,
    "book": 12 / 100,
    "She": 17 / 100,
    "cat": 2 / 100,
}

# 计算 PPL
log_p = 0  # 计算对数化概率
total_words = 0  # 统计token 数

for sent in sentences:
    for word in sent:
        log_p += math.log(unigram[word])
        total_words += 1

ppl = math.exp(-1 / total_words * log_p)
print()
print(f"最终计算的 PPL 为：{ppl}")
