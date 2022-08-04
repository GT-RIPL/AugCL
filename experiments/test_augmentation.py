import argparse
import src.augmentations as augmentations
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--aug_key", default="splice_color", type=str)
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
    dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.sample_png_folder),
        batch_size=3,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    data_iter = iter(dataloader)
    x = data_iter.next()
    augs_tnsr = augmentations.aug_to_func[args.aug_key](x) / 255.0
    show_stacked_imgs(augs_tnsr.cpu().numpy())
    plt.savefig(args.save_file_name)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
