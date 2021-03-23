import torch

# khởi tạo ma trận rỗng
x = torch.empty(5, 3)
print(x)

# khởi tạo ma trận ngẫu nhiên
x = torch.rand(2, 3)
print(x)

# khởi tạo ma trận 0
x = torch.zeros(3, 2)
print(x)

# khởi tạo ma trận từ list
x = torch.tensor([1, 2, 3])
print(x)

# khởi tạo ma trận tương tự như ma trận khác về kích thước và có thể override dtype
# bất cứ cách khởi tạo trên nào cũng có thể áp dựng dược trừ tensor
# empty_likem, rand_like, zeros_like
x = torch.rand_like(x, dtype=torch.double)
print(x)

# get size của ma trận
print(x.size())