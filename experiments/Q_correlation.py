import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_mode,
        intensity=args.train_distracting_cs_intensity,
    )

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps + 1,
        batch_size=args.batch_size,
    )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    done = True
    for _ in range(args.num_step_samples):
        if done:
            obs = env.reset()
            done = False

        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs


if __name__ == "__main__":
    args = parse_args()
    main(args)
