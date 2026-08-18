def n_gram_fn(n_gram):
    # 1- 测试数据
    word_list = ["aa", "bb", "cc", "dd"]

    # 2- 对相邻的词进行合并
    """
        2-gram：
            ["aa","bb","cc","dd"]
            ["bb","cc","dd"]

        3-gram：
            ["aa","bb","cc","dd"]
            ["bb","cc","dd"]
            ["cc","dd"]
    """
    lists = [word_list[i:] for i in range(n_gram)]
    print(list(zip(*lists)))


if __name__ == "__main__":
    n_gram_fn(n_gram=1)
    print("-" * 50)
    n_gram_fn(n_gram=2)
    print("-" * 50)
    n_gram_fn(n_gram=3)
    print("-" * 50)
