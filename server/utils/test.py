import torch
print(torch.__version__)  # Should print a version with 'cu118'
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should return 11.8
print(torch.cuda.device_count())  # Should be > 0
print(torch.cuda.get_device_name(0))  # Should print GPU name
