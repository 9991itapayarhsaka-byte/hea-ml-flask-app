import torch
import time

device = torch.device("cuda")

print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# Large matrices to stress GPU
x = torch.randn(8000, 8000, device=device)
y = torch.randn(8000, 8000, device=device)

torch.cuda.synchronize()
start = time.time()

for i in range(10):
    z = torch.matmul(x, y)

torch.cuda.synchronize()
end = time.time()

print("Total Time:", end - start, "seconds")
