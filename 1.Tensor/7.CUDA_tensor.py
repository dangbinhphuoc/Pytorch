import torch

x = torch.tensor(range(0, 3))

# kiểm tra có tồn tại CUDA không.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    # khởi tạo một cuda device object
    device = torch.device("cuda") 
    # trực tiếp khởi tạo một tensor trên GPU
    y_gpu = torch.ones(3, 3, device=device) 
    # truyền giá trị tensor vào GPU
    x_gpu = x.to(device) 
    # thực hiện phép tính trên GPU
    z_gpu = x_gpu + y_gpu 
    print(z)
    # mỗi tensor chỉ được lưu trữ trên 1 GPU nên lưu trữ theo index
    x_gpu = x.to(device='cuda:0')
    # hoặc
    x_gpu = x.cuda(0)
    # chuyển tensor từ GPU thành CPU
    print(z.to("cpu", torch.double)) 
    # hoặc
    x_cpu = x_gpu.cpu()


