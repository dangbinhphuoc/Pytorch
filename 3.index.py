import torch

arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
# truy cập index
print(x[:, 1]) # lấy phần tử cột 2
print(x[1, :]) # lấy phần tử dòng 2
print(x[1, 1]) # lấy phần tử dòng 2 cột 2

print()

x = torch.tensor(range(0, 10))
print(x)
print(x.shape)
# hỗ trợ index âm
print(x[-1]) # x[-1] = x[x.shape[0]-1]
# cú pháp sliding là: start:stop:step
print(x[0:10:3])