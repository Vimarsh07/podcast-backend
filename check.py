import torch

print(torch.__version__)
print(torch.cuda.is_available())  # Should now return True
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Your GPU name
