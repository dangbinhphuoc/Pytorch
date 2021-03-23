import torch

x = torch.tensor(range(0, 3))

# kiểm tra có tồn tại CUDA không.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda") # khởi tạo một cuda device object
    y = torch.ones(3, 3, device=device) # trực tiếp khởi tạo một tensor trên GPU
    x = x.to(device) # truyền giá trị tensor vào thiết bị 
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
