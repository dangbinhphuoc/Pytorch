import torch    

# khi tạo graph các non-leaf node (tensor) được cấp phát bộ nhớ
# context (ctx) được tạo để lưu các biến tạm cho quá trình backward. 
# sau đó, khi gọi backward, thì đạo hàm được tính ngược lại cho các leaf tensor
# và graph bị hủy, vùng nhớ lưu các non-leaf node, biến tạm trong context được giải phóng. 
# do đó không thể backward 2 lần liên tiếp.

x = torch.tensor([1., 2., 3.], requires_grad=True) # graph chưa được tạo
y = 2*x + 1 # bắt đầu tạo graph khi chạy qua dòng này
z = sum(y)
z.backward()
print(x.grad) # tensor([2., 2., 2.])
# z.backward() # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling .backward() or autograd.grad() the first time.

# tuy nhiên khi train model DL
# cần train nhiều epoch
# mỗi epoch lại có nhiều step
# nên cần gọi backward nhiều lần để tính đạo hàm ngược lại
# để thực hiện backward nhiều lần mình cần để thuộc tính retain_graph = True
# tuy nhiên khi backward nhiều lần thì đạo hàm sẽ cộng dồn vào leaf tensor.
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2*x + 1
z = sum(y)
z.backward(retain_graph=True)
print(x.grad) # 2, 2, 2
z.backward(retain_graph=True)
print(x.grad) # 4, 4, 4
z.backward(retain_graph=True)
print(x.grad) # 6, 6, 6
# vì vậy khi optimizer chúng ta nên dùng zero_grad

# from torch.autograd import Variable
# import torch.optim as optim

# def linear_model(x, W, b):
#     return torch.matmul(x, W) + b

# data, targets = ...

# W = Variable(torch.randn(4, 3), requires_grad=True)
# b = Variable(torch.randn(3), requires_grad=True)

# optimizer = optim.Adam([W, b])

# for sample, target in zip(data, targets):
#     # clear out the gradients of all Variables 
#     # in this optimizer (i.e. W, b)
#     optimizer.zero_grad()
#     output = linear_model(sample, W, b)
#     loss = (output - target) ** 2
#     loss.backward()
#     optimizer.step()