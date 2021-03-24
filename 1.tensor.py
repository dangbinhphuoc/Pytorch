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

# nơi lưu trữ
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.storage())

# contiguous tensors
x_t = x.t()
print(x)
print(x_t)
print(id(x.storage()) == id(x_t.storage()))
print(x.stride()) # (1, 3)
print(x_t.stride()) # (3, 1)

# kiểm tra lưu trữ có liền kề không do storage lưu trữ theo kiểu liền kề nhau
# contiguous tensors
print(x.is_contiguous())
print(x_t.is_contiguous())
print(x.view(1, -1))
print(x_t.view(1, -1)) # RuntimeError