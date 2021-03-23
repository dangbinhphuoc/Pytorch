import torch

arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
# truy cập index
print(x[:, 1]) # lấy phần tử cột 2
print(x[1, :]) # lấy phần tử dòng 2
print(x[1, 1]) # lấy phần tử dòng 2 cột 2