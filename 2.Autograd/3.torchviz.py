from torchviz import make_dot
import torch

# def forward(ctx, input): nhận các tensor inputs, và trả về tensor output. 
#                          ctx để lưu lại các tensor cần thiết trong quá trình backward (chain rule).
# def backward(ctx, grad_output): grad_output chứa đạo hàm của loss đến tensor ở node đấy.
#                                 ctx lấy các giá trị lưu ở hàm forward để tính đạo hàm ngược qua node đó.

# custom hàm bình phương
class MySquare(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input**2

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return 2*input*grad_output

# alias để gọi hàm
my_square = MySquare.apply

# xây graph
x = torch.tensor([3])
y = torch.tensor([10])
a = torch.tensor([1.], requires_grad=True)
b = torch.tensor([2.], requires_grad=True)

y_hat = a*x + b
z = y_hat - y
L = my_square(z)
print(make_dot(L))

L.backward()
print(a.grad) # -30
print(b.grad) # -10

# def backward (ctx, grad_output):
#         # thuộc tính grad chỉ cần thiết nếu ở leaf tensor.
# 	self.Tensor.grad = grad_output

#         # duyệt qua các input đến node này để trả đạo hàm ngược lại.
# 	for inp in self.inputs:
# 		if inp.grad_fn is not None:
#                         # local_grad là đạo hàm trong node đang tính
# 			new_incoming_gradients = grad_output * local_grad(self.Tensor, inp)
			
# 			inp.grad_fn.backward(new_incoming_gradients)
# 		else:
# 			pass