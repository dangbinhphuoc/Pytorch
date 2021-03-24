import torch

x = torch.randn(3, requires_grad = True)
yhat = torch.randn(3, requires_grad = True)*2
y = x*2
l = ((y-yhat)**2).mean()
print(l)

# Khởi tạo một vector gradient tự do v
v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float)
# Tính ma trận Jacobian (đạo hàm của y theo v)
l.backward(v)
# Tính tích vector-jacobian chính là đạo hàm của 
print(x.grad)