import torch

arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
print(x)

y = torch.ones_like(x)
print(y)

# phép cộng (+)
z = x + y
print(z)
z = torch.add(x, y)
print(z)
torch.add(x, y, out=z)
print(z)

# lưu kết quả ngay trên đối tượng được áp dụng
y.add_(x)
print(y)

print()

# phép trừ (-)
arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
print(x)

y = torch.ones_like(x)
print(y)

z = x - y
print(z)
z = torch.sub(x, y)
print(z)
torch.sub(x, y, out=z)
print(z)

print()

# phép chia (/)
arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
print(x)

y = torch.ones_like(x)
print(y)

z = y / x
print(z)
z = torch.div(y, x)
print(z)
torch.div(y, x, out=z)
print(z)

print()

# phép nhân (*)
arr = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(arr)
print(x)

y = torch.ones_like(x)
print(y)

z = x * y
print(z)
z = torch.mul(x, y)
print(z)
torch.mul(x, y, out=z)
print(z)