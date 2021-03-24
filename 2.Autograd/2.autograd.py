import torch

x = torch.ones(2, 2, requires_grad=True)
# hoặc
# x = torch.ones(2, 2)
# x.requires_grad_(True)
print(x)

# y = x + 2
y = x + 2
print(y)
# .grad_fn được dùng để theo dõi quá trình tính toán
print(y.grad_fn)

# z = 3*(x+2)^2
z = y*y*3
print(z)

# out = z/4 = (3/4)*(x+2)^2
out = z.mean()
print(out)

# d(out)/dx = (3/2)*(x+2)
# thực hiện backprop bằng lệnh .backward()
out.backward()
# kết quả đạo hàm sẽ được lưu tại .grad
print(x.grad)

from torchviz import make_dot
print(make_dot(out))

    