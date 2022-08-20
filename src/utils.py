import argparse
from PIL import Image
import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = "Input shall be a HxWx3 ndarray"
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, "RGB")
    return img


def make_dir(dir_path, exist_ok=True):
    os.makedirs(dir_path, exist_ok=exist_ok)
    return dir_path


def dump_args_json(args: dict, log_dir: str, model_dir: str):
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json_dict = vars(args)
        json_dict["checkpoint_dir"] = model_dir
        json.dump(json_dict, f, sort_keys=True, indent=4)


def args_json_to_dot_dict(args_file_path: str):
    config_dict = json.load(open(args_file_path))
    return dotdict(config_dict)


def config_json_to_args(args, config_path):
    config_dict = json.load(open(config_path))
    for key, value in config_dict.items():
        args.__dict__[key] = value

    return args


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def assert_params_matching(net, net_2):
    for param, param_2 in zip(net.parameters(), net_2.parameters()):
        assert torch.all(param.data == param_2.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        "timestamp": str(datetime.now()),
        "git": subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode(),
        "args": vars(args),
    }
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def load_config(key=None):
    path = os.path.join("setup", "config.cfg")
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def listdir(dir_path, filetype="jpg", sort=True):
    fpath = os.path.join(dir_path, f"*.{filetype}")
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def save(self, save_dir):
        # if self.idx == self.last_save:
        #     return
        # path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        # payload = [
        #     self._obses[self.last_save : self.idx],
        #     self.actions[self.last_save : self.idx],
        #     self.rewards[self.last_save : self.idx],
        #     self.not_dones[self.last_save : self.idx],
        # ]
        # self.last_save = self.idx
        # torch.save(payload, path)

        path = os.path.join(save_dir, "buffer.pt")
        payload = [self._obses, self.actions, self.rewards, self.not_dones, self.idx]
        torch.save(payload, path)

    def load(self, save_dir, end_step=None):
        # chunks = os.listdir(save_dir)
        # chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        # for chunk in chucks:
        #     start, end = [int(x) for x in chunk.split(".")[0].split("_")]
        #     path = os.path.join(save_dir, chunk)
        #     payload = torch.load(path)
        #     assert self.idx == start
        #     self._obses[start:end] = payload[0]
        #     self.actions[start:end] = payload[1]
        #     self.rewards[start:end] = payload[2]
        #     self.not_dones[start:end] = payload[3]
        #     self.idx = end

        #     if end == end_step:
        #         break
        payload = torch.load(os.path.join(save_dir, "buffer.pt"))
        self._obses = payload[0]
        self.actions = payload[1]
        self.rewards = payload[2]
        self.not_dones = payload[3]
        self.idx = payload[4]

    def tensor_buffer_samples(self, idxs):
        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, next_obs, actions, rewards, not_dones

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = obses
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(0, self.capacity if self.full else self.idx, size=n)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_obs(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def sample_curl(self, n=None):
        idxs = self._get_idxs(n)

        (obs, next_obs, actions, rewards, not_dones) = self.tensor_buffer_samples(
            idxs=idxs
        )

        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        idxs = self._get_idxs(n)

        (obs, next_obs, actions, rewards, not_dones) = self.tensor_buffer_samples(
            idxs=idxs
        )

        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None, idxs=None):
        if idxs is None:
            idxs = self._get_idxs(n)

        (obs, next_obs, actions, rewards, not_dones) = self.tensor_buffer_samples(
            idxs=idxs
        )

        return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"
