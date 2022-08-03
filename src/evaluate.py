import time
import torch
import pandas as pd
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_eval_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations


def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
    episode_rewards = []
    for i in tqdm(range(num_episodes)):
        if adapt:
            ep_agent = deepcopy(agent)
            ep_agent.init_pad_optimizer()
        else:
            ep_agent = agent
        obs = env.reset()
        video.init(enabled=True)
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            video.record(env, eval_mode)
            episode_reward += reward
            if adapt:
                ep_agent.update_inverse_dynamics(
                    *augmentations.prepare_pad_batch(obs, next_obs, action)
                )
            obs = next_obs

        video.save(f"eval_{eval_mode}_{i}.mp4")
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def confirm_model_results(work_dir: str, train_dot_dict, agent):
    env = make_env(
        domain_name=train_dot_dict.domain_name,
        task_name=train_dot_dict.task_name,
        seed=train_dot_dict.seed + 42,
        episode_length=train_dot_dict.episode_length,
        action_repeat=train_dot_dict.action_repeat,
        image_size=train_dot_dict.image_size,
    )

    mean_ep_reward = evaluate(
        env=env,
        agent=agent,
        video=VideoRecorder(None, height=448, width=448),
        num_episodes=5,
        eval_mode="train",
    )

    try:
        df = pd.read_csv(os.path.join(work_dir, "eval.csv"))
        last_update_reward_mu = df["episode_reward"].iloc[-1]

        reward_diff = mean_ep_reward - last_update_reward_mu
        allowed_diff = -50
        assert (
            reward_diff > allowed_diff
        ), f"Difference between current eval and last known eval is less than {allowed_diff}: {reward_diff}"
    except:
        pass


def main(args):
    train_dot_dict = utils.args_json_to_dot_dict(
        args_file_path=os.path.join(args.dir_path, "args.json")
    )
    # Set seed
    utils.set_seed_everywhere(train_dot_dict.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=train_dot_dict.domain_name,
        task_name=train_dot_dict.task_name,
        seed=train_dot_dict.seed + 42,
        episode_length=train_dot_dict.episode_length,
        action_repeat=train_dot_dict.action_repeat,
        image_size=train_dot_dict.image_size,
        mode=args.env_mode,
        intensity=args.distracting_cs_intensity,
    )

    # Set working directory
    work_dir = args.dir_path
    checkpt_path = os.path.join(
        train_dot_dict.checkpoint_dir, str(train_dot_dict.train_steps) + ".pt"
    )
    print("Working directory:", work_dir)

    assert os.path.exists(
        checkpt_path
    ), f"Checkpoint {str(train_dot_dict.train_steps)}.pt does not exists"
    assert os.path.exists(work_dir), "specified working directory does not exist"
    assert os.path.exists(
        train_dot_dict.checkpoint_dir
    ), "specified checkpoint directory does not exist"

    # TODO: VideoRecorder disabled
    # video_dir = utils.make_dir(os.path.join(work_dir, "video"), exist_ok=True)
    video = VideoRecorder(None, height=448, width=448)

    results_fp = os.path.join(work_dir, "evaluate.csv")

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    cropped_obs_shape = (
        3 * train_dot_dict.frame_stack,
        train_dot_dict.image_crop_size,
        train_dot_dict.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=train_dot_dict,
    )

    agent = torch.load(checkpt_path)
    agent.train(False)

    confirm_model_results(work_dir=work_dir, train_dot_dict=train_dot_dict, agent=agent)

    print(
        f"\nEvaluating {work_dir} for {args.num_episodes} episodes (mode: {train_dot_dict.env_mode})"
    )
    reward = evaluate(env, agent, video, args.num_episodes, train_dot_dict.env_mode)
    print("Reward:", int(reward))

    adapt_reward = None
    if train_dot_dict.algorithm == "pad":
        env = make_env(
            domain_name=train_dot_dict.domain_name,
            task_name=train_dot_dict.task_name,
            seed=train_dot_dict.seed + 42,
            episode_length=train_dot_dict.episode_length,
            action_repeat=train_dot_dict.action_repeat,
            mode=train_dot_dict.eval_mode,
            intensity=args.distracting_cs_intensity,
        )
        adapt_reward = evaluate(
            env,
            agent,
            video,
            train_dot_dict.eval_episodes,
            train_dot_dict.eval_mode,
            adapt=True,
        )
        print("Adapt reward:", int(adapt_reward))

    # Save results
    data = [
        [
            time.strftime("%m-%d-%y", time.gmtime()),
            args.env_mode,
            args.distracting_cs_intensity,
            args.num_episodes,
            reward,
            train_dot_dict.train_steps,
            adapt_reward,
        ]
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "date",
            "env mode",
            "distracting intensity",
            "num episodes",
            "reward",
            "train_steps",
            "adapt reward",
        ],
    )
    if os.path.exists(results_fp):
        df = pd.read_csv(results_fp).append(df)
    else:
        try:
            eval_last_row = (
                pd.read_csv(os.path.join(work_dir, "eval.csv"))
                .iloc[-1]
                .filter(regex="episode_reward")
            )
            eval_keys = eval_last_row.keys()
            train_reward = eval_last_row["episode_reward"]
            eval_reward = eval_last_row[eval_keys[1]]
            test_env_name = eval_keys[1].replace("episode_reward_", "")
            intensity = 0.0

            if "distracting_cs" in test_env_name:
                intensity = test_env_name.replace("distracting_cs_", "")
                test_env_name = "distracting_cs"

            eval_df = pd.DataFrame(
                [
                    [
                        None,
                        "train",
                        0.0,
                        train_dot_dict.eval_episodes_final_step,
                        train_reward,
                        train_dot_dict.train_steps,
                        None,
                    ],
                    [
                        None,
                        test_env_name,
                        intensity,
                        train_dot_dict.eval_episodes_final_step,
                        eval_reward,
                        train_dot_dict.train_steps,
                        None,
                    ],
                ],
                columns=[
                    "date",
                    "env mode",
                    "distracting intensity",
                    "num episodes",
                    "reward",
                    "train_steps",
                    "adapt reward",
                ],
            )
            df = df.append(eval_df)
        except:
            pass

    df.to_csv(results_fp)
    print("Saved results to", results_fp)


if __name__ == "__main__":
    args = parse_eval_args()
    main(args)
