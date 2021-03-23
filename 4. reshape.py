import torch

arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)

#reshape
y = x.view(3, 2)
z = x.view(6, 1)
print(x.size(), y.size(), z.size())

#reshape tự định dạng
t = x.view(-1, 6) # -1 sẽ tự tính nếu chúng ta không cung cấp dữ liệu
print(t.size())
