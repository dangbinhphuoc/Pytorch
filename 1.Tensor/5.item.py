import torch

# only one element tensors can be converted to Python scalars
# arr = [[1, 2, 3], [4, 5, 6]]
# x = torch.tensor(arr)
# i = x.item()
# print(i)

x = torch.tensor([1.2])
i = x.item()
print(i)