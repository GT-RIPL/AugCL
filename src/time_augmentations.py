import torch
import augmentations
import time

input = torch.rand(128, 9, 84, 84).to("cuda")

for key, value in augmentations.aug_to_func.items():
    print(key)
    if "overlay" in key or "splice" in key:
        value(input)
    start_time = time.time()
    value(input)
    print(f"Time: {time.time() - start_time}")
