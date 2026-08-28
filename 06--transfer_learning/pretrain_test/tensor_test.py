import torch

if __name__ == "__main__":
    pred_index = torch.tensor([[11, 22, 33], [44, 55, 66]])
    labels = torch.tensor([[11, 888, 33], [999, 55, 66]])

    print("-" * 40)
    result_1 = pred_index == labels
    print(result_1)

    print("-" * 40)
    result_2 = (pred_index == labels).sum()
    print(result_2)

    print("-" * 40)
    result_3 = (pred_index == labels).sum().item()
    print(result_3)
