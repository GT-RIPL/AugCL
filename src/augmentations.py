import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os


dataloader = None
data_iter = None


def _load_data(
    sub_path: str, batch_size: int = 256, image_size: int = 84, num_workers: int = 16
):
    global data_iter, dataloader
    for data_dir in utils.load_config("datasets"):
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, sub_path)
            if not os.path.exists(fp):
                print(f"Warning: path {fp} does not exist, falling back to {data_dir}")
            dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    fp,
                    TF.Compose(
                        [
                            TF.RandomResizedCrop(image_size),
                            TF.RandomHorizontalFlip(),
                            TF.ToTensor(),
                        ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            data_iter = iter(dataloader)
            break
    if data_iter is None:
        raise FileNotFoundError(
            "failed to find image data at any of the specified paths"
        )
    print("Loaded dataset from", data_dir)


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
    partition = "val" if use_val else "train"
    sub_path = os.path.join("places365_standard", partition)
    print(f"Loading {partition} partition of places365_standard...")
    _load_data(
        sub_path=sub_path,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )


def _load_coco(batch_size=256, image_size=84, num_workers=16, use_val=False):
    sub_path = "COCO"
    print(f"Loading COCO 2017 Val...")
    _load_data(
        sub_path=sub_path,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )


def _get_data_batch(batch_size):
    global data_iter
    try:
        imgs, _ = next(data_iter)
        if imgs.size(0) < batch_size:
            data_iter = iter(dataloader)
            imgs, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        imgs, _ = next(data_iter)
    return imgs.cuda()


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3

    imgs = imgs.view([b, frames, 3, h, w])
    imgs = (
        imgs[:, :, 0, ...] * 0.2989
        + imgs[:, :, 1, ...] * 0.587
        + imgs[:, :, 2, ...] * 0.114
    )

    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(
        device
    )  # broadcast tiling
    return imgs


def random_grayscale(images, p=0.3):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: cpu or cuda
    returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.0
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.0
    return out


def random_flip(images, p=0.2):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: cpu or gpu,
    p: prob of applying aug,
    returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape

    images = images.to(device)

    flipped_images = images.flip([3])

    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1]  # // 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)

    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]

    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out


def random_rotation(images, p=0.3):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: str, cpu or gpu,
    p: float, prob of applying aug,
    returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape

    images = images.to(device)

    rot90_images = images.rot90(1, [2, 3])
    rot180_images = images.rot90(2, [2, 3])
    rot270_images = images.rot90(3, [2, 3])

    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)

    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[torch.where(mask == i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(
            device
        )
        m = m[:, :, None, None]
        masks[i] = m

    out = (
        masks[0] * images
        + masks[1] * rot90_images
        + masks[2] * rot180_images
        + masks[3] * rot270_images
    )

    out = out.view([bs, -1, h, w])
    return out


def kornia_color_jitter(imgs, bright=0.4, contrast=0.4, satur=0.4, hue=0.49, p=1):
    """
    inputs np array outputs tensor
    """
    b, c, h, w = imgs.shape
    num_frames = int(c / 3)
    num_samples = int(p * b * num_frames)

    sampled_idxs = torch.from_numpy(np.random.randint(0, b, num_samples))
    imgs = imgs.view(-1, 3, h, w)
    model = kornia.augmentation.ColorJitter(
        brightness=bright, contrast=contrast, saturation=satur, hue=hue
    )
    sampled_imgs = imgs[sampled_idxs]
    imgs[sampled_idxs] = model(sampled_imgs)
    return imgs.view(b, c, h, w)


def load_dataloader(batch_size, image_size, dataset="coco"):
    if dataset == "places365_standard":
        if dataloader is None:
            _load_places(batch_size=batch_size, image_size=image_size)
    elif dataset == "coco":
        if dataloader is None:
            _load_coco(batch_size=batch_size, image_size=image_size)
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )


def random_overlay(x, dataset="coco"):
    """Randomly overlay an image from Places or COCO"""
    global data_iter
    alpha = 0.5

    load_dataloader(batch_size=x.size(0), image_size=x.size(-1), dataset=dataset)

    imgs = _get_data_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)

    return ((1 - alpha) * (x / 255.0) + (alpha) * imgs) * 255.0


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)


def splice(x, R_threshold=0.35, G_threshold=0.45, B_threshold=1):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    weight = torch.ones(x_rgb.shape)
    weight[:, 0] = weight[:, 0] * R_threshold
    weight[:, 1] = weight[:, 1] * G_threshold
    weight[:, 2] = weight[:, 2] * B_threshold
    mask = x_rgb > weight.to(x.get_device())
    out = (mask * x_rgb + (~mask) * overlay) * 255.0
    return out.reshape(n, c, h, w)


def splice_color(x, R_threshold=0.35, G_threshold=0.45, B_threshold=1):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    weight = torch.ones(x_rgb.shape)
    weight[:, 0] = weight[:, 0] * R_threshold
    weight[:, 1] = weight[:, 1] * G_threshold
    weight[:, 2] = weight[:, 2] * B_threshold
    mask = x_rgb > weight.to(x.get_device())
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights))
        conv_out = out if i == 0 else torch.cat([conv_out, out], axis=0)
    out = (mask * conv_out + (~mask) * overlay) * 255.0
    return out.reshape(n, c, h, w)


def batch_from_obs(obs, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
    """Prepare batch for self-supervised policy adaptation at test-time"""
    batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
    batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
    batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

    return random_crop(batch_obs), random_crop(batch_next_obs), batch_action


def identity(x):
    return x


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"
    assert isinstance(x, torch.Tensor) and x.is_cuda, "input must be CUDA tensor"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


aug_to_func = {
    "grayscale": random_grayscale,
    "flip": random_flip,
    "rotate": random_rotation,
    "rand_conv": random_conv,
    "shift": random_shift,
    "color_jitter": kornia_color_jitter,
    "identity": identity,
    "overlay": random_overlay,
    "splice": splice,
    "splice_color": splice_color,
    "crop": random_crop,
}
