import torch
import numpy as np

# chuyển tensor thành numpy
x = torch.tensor([1,2,3])
x_np = x.numpy()

# Torch tensor và Numpy array sẽ dùng chung vùng nhớ 
# nên khi thay đổi torch tensor thì Numpy array cũng thay đổi
x[1] = 0
print(x) # output: [1, 0, 3]
print(x_np) # output: [1, 0, 3]

# Nếu tensor được lưu trên GPU thì không thể chuyển trực tiếp tensor sang Numpy array được,
# Cần copy nội dung của tensor sang CPU trước rồi mới chuyển sang Numpy array. 
# Do đó 2 biến trên gpu và np không dùng chung vùng nhớ và sửa 1 biến không ảnh hưởng biến còn lại.
x_gpu = torch.tensor([1, 2, 3], device='cuda')
x_np = x_gpu.numpy() # Error
x_np = x_gpu.cpu().numpy() # ok
x_gpu[1] = 0 
print(x_gpu) # output: [1, 0, 3]
print(x_np) # output: [1, 2, 3]

# chuyển từ numpy sang tensor
x_np = np.array([1, 2, 3])
x_cpu = torch.from_numpy(x_np)