import argparse
import numpy as np
import augmentations
import torch
import os
from os import listdir
from os.path import isfile, join
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--aug_key", default="thresholded_overlay_color", type=str)
    parser.add_argument("--save_file_name", default="aug_test.png", type=str)
    parser.add_argument(
        "--sample_png_folder",
        default="samples/walker/walk/distracting_cs/0.1",
        type=str,
    )

    return parser.parse_args()


def show_imgs(x, max_display=12):
    grid = (
        make_grid(torch.from_numpy(x[:max_display]), 4).permute(1, 2, 0).cpu().numpy()
    )
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)


def show_stacked_imgs(x, max_display=12):
    fig = plt.figure(figsize=(12, 12))
    stack = 3

    for i in range(1, stack + 1):
        grid = (
            make_grid(torch.from_numpy(x[:max_display, (i - 1) * 3 : i * 3, ...]), 4)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        fig.add_subplot(1, stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title("frame " + str(i))
        plt.imshow(grid)


def main(args):
    tnsr_list = list()
    sample_imgs_files = [
        os.path.join(args.sample_png_folder, f)
        for f in listdir(args.sample_png_folder)
        if isfile(join(args.sample_png_folder, f))
    ]

    for fp in sample_imgs_files:
        img = Image.open(fp=fp)
        img_tnsr = transforms.ToTensor()(img)
        tnsr_list.append(img_tnsr)

    imgs_tnsr = torch.unsqueeze(torch.cat(tnsr_list, dim=0), dim=0) * 255
    augs_tnsr = augmentations.aug_to_func[args.aug_key](imgs_tnsr.to("cuda")) / 255.0
    show_stacked_imgs(augs_tnsr.cpu().numpy())
    plt.savefig(args.save_file_name)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
