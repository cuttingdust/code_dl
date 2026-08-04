import torch

if __name__ == "__main__":
    data = torch.tensor(data=[11])
    print(data)

    # 张量中不能放非数字：例如字符串
    # data2 = torch.tensor(data=["hello"])
    # print(data2)
