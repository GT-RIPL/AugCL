import copy
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
from train import evaluate


def main(args):
    train_dot_dict = utils.args_json_to_dot_dict(
        args_file_path=os.path.join(args.dir_path, "train_dot_dict.json")
    )
    # Set seed
    utils.set_seed_everywhere(train_dot_dict.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=train_dot_dict.domain_name,
        task_name=train_dot_dict.task_name,
        seed=train_dot_dict.seed,
        episode_length=train_dot_dict.episode_length,
        action_repeat=train_dot_dict.action_repeat,
        image_size=train_dot_dict.image_size,
        mode=train_dot_dict.train_mode,
        intensity=train_dot_dict.train_distracting_cs_intensity,
    )
    test_env = (
        make_env(
            domain_name=train_dot_dict.domain_name,
            task_name=train_dot_dict.task_name,
            seed=train_dot_dict.seed + 42,
            episode_length=train_dot_dict.episode_length,
            action_repeat=train_dot_dict.action_repeat,
            image_size=train_dot_dict.image_size,
            mode=train_dot_dict.eval_mode,
            intensity=train_dot_dict.distracting_cs_intensity,
        )
        if train_dot_dict.eval_mode is not None
        else None
    )

    # Create working directory
    work_dir = os.path.join(
        train_dot_dict.log_dir,
        train_dot_dict.domain_name + "_" + train_dot_dict.task_name,
        train_dot_dict.algorithm,
        train_dot_dict.id,
        "seed_" + str(train_dot_dict.seed),
    )
    print("Working directory:", work_dir)
    train_dot_dict.__dict__["train_date"] = time.strftime("%m-%d-%y", time.gmtime())

    if not train_dot_dict.test_code_mode:
        assert not os.path.exists(
            os.path.join(work_dir, "train.log")
        ), "Specified working directory has existing train.log. Ending program."
    os.makedirs(work_dir, exist_ok=True)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))

    if train_dot_dict.save_video:
        video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(
        video_dir if train_dot_dict.save_video else None, height=448, width=448
    )

    utils.write_info(args, os.path.join(work_dir, "info.log"))
    utils.dump_args_json(args=args, log_dir=work_dir, model_dir=model_dir)

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=(train_dot_dict.train_steps + args.train_steps),
        batch_size=train_dot_dict.batch_size,
    )
    cropped_obs_shape = (
        3 * train_dot_dict.frame_stack,
        train_dot_dict.image_crop_size,
        train_dot_dict.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    pretrained_agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=train_dot_dict,
    )
    agent_args = copy.deepcopy(train_dot_dict)
    agent_args.algorithm = args.algorithm
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=agent_args,
    )

    utils.soft_update_params(pretrained_agent.actor, agent.actor, 1)
    utils.soft_update_params(pretrained_agent.critic, agent.critic, 1)
    utils.soft_update_params(pretrained_agent.critic_target, agent.critic_target, 1)

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(
        train_dot_dict.train_steps, args.train_steps + train_dot_dict.train_steps + 1
    ):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % train_dot_dict.eval_freq == 0:
                num_episodes = (
                    train_dot_dict.eval_episodes_final_step
                    if step == train_dot_dict.train_steps
                    else train_dot_dict.eval_episodes
                )
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(
                    env,
                    agent,
                    video,
                    num_episodes,
                    L,
                    step,
                    train_dot_dict=train_dot_dict,
                )
                if test_env is not None:
                    evaluate(
                        test_env,
                        agent,
                        video,
                        num_episodes,
                        L,
                        step,
                        test_env=True,
                        train_dot_dict=train_dot_dict,
                    )
                L.dump(step)

            # Save agent periodically
            if (
                step > start_step
                and step % train_dot_dict.save_freq == 0
                or step == train_dot_dict.train_steps - 1
            ):
                torch.save(agent, os.path.join(model_dir, f"{step}.pt"))

            L.log("train/episode_reward", episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

        # Sample action for data collection
        if step < train_dot_dict.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= train_dot_dict.init_steps:
            num_updates = (
                train_dot_dict.init_steps if step == train_dot_dict.init_steps else 1
            )
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print("Completed training for", work_dir)


if __name__ == "__main__":
    train_dot_dict = parse_args()
    main(train_dot_dict)
