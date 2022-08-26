import numpy as np
import torch
import augmentations as augmentations
import time

input = torch.rand(128, 9, 84, 84).to("cuda").detach()
dataloader_booted_up = False
for key, value in augmentations.aug_to_func.items():
    print(key)

    # This is just to bootup dataloader
    if not dataloader_booted_up and "overlay" in key or "splice" in key:
        value(input)
        dataloader_booted_up = True

    time_list = list()
    for i in range(10):
        start_time = time.time()
        value(input)
        time_list.append(time.time() - start_time)

    print(f"{key}. Mean: {np.mean(time_list)}, STD: {np.std(time_list)}")
