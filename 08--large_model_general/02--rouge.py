from rouge import Rouge

# 生成文本
generated_text = "it is a nice day today"

# 参考文本列表
reference_texts = ["today is a nice day"]

# 计算 ROUGE 指标
rouge = Rouge()
scores = rouge.get_scores(generated_text, reference_texts[0])

print()
# 打印结果
print(f"ROUGE-1 precision: {scores[0]['rouge-1']['p']}")
print(f"ROUGE-1 recall: {scores[0]['rouge-1']['r']}")
print(f"ROUGE-1 F1 score: {scores[0]["rouge-1"]["f"]}")

# ROUGE-1 precision: 0.8333333333333334
# ROUGE-1 recall: 1.0
# ROUGE-1 F1 score: 0.9090909041322315

print("全部结果：", scores)
# rouge-l 计算最长公共子序列
