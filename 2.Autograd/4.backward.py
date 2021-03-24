import torch

# x = torch.tensor([1., 2., 3.], requires_grad=True)
# y = 2*x + 1
# y.backward() # RuntimeError: grad can be implicitly created only for scalar outputs

# .backward() chỉ thực hiện trên 1 số thực.
# tuy nhiên cũng hợp lý vì loss funtion chỉ có 1 số.
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2*x + 1
z = sum(y)
z.backward()

print(x.grad) # 2, 2, 2

# khi gọi y.backward truyền 1 vector tensor có kích thước bằng kích thước của y, ý nghĩa có thể hiểu đó chính là đạo hàm của loss với y.

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2*x + 1
y.backward(gradient=torch.tensor([1, 2, 1]))

print(x.grad) # 2, 4, 2