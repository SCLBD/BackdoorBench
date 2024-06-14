import torch
 
print(torch.cuda.is_available())
#cuda是否可用；
 
print(torch.cuda.device_count())
#返回gpu数量；
 
print(torch.cuda.get_device_name())
#返回gpu名字，设备索引默认从0开始；
 
print(torch.cuda.current_device())
#返回当前设备索引；
for i in range(torch.cuda.device_count()):
    sync_device = torch.device('cuda')
    print(sync_device)
    z = torch.empty([4, 512], device=sync_device)
    print(z.device)

sync_device_out = torch.device('cuda', 2)
print(sync_device_out)
z = torch.empty([4, 512], device=sync_device_out)
print(z.device)