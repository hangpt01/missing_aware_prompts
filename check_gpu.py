# from pytorch_lightning import Trainer
# from pytorch_lightning.utilities.device_parser import parse_gpu_ids

# # Kiểm tra GPU nào đang được nhận diện
# gpu_ids = parse_gpu_ids([2,3])
# print(gpu_ids)

import torch

# Kiểm tra tổng số GPU
print(f"Total GPUs available: {torch.cuda.device_count()}")

# Kiểm tra GPU 2 và 3
for i in [2, 3]:
    if i < torch.cuda.device_count():
        device = torch.device(f"cuda:{i}")
        print(f"GPU {i} is available: {torch.cuda.is_available()}")
        print(f"GPU {i} name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"GPU {i} is not available.")