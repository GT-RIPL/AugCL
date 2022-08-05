import pandas as pd
import torch
import os
import gym
import utils as utils
from tqdm import tqdm
from arguments import parse_Q_variance_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from evaluate import confirm_model_results
import augmentations as augmentations


def create_aug_var_dict():
    q_var_dict = dict()

    for key in augmentations.aug_to_func.keys():
        q_var_dict[key] = {"Q1": [], "Q2": [], "Q1-target": [], "Q2-target": []}

    return q_var_dict


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
        mode="train",
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

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.num_samples,
        batch_size=args.num_samples,
    )

    agent = torch.load(checkpt_path)
    agent.train(False)

    confirm_model_results(work_dir=work_dir, train_dot_dict=train_dot_dict, agent=agent)
    q_var_dict = create_aug_var_dict()

    done = True
    for _ in tqdm(range(args.num_samples)):
        if done:
            obs = env.reset()
        with utils.eval_mode(agent):
            action = agent.select_action(obs)

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

    obs, actions, _, _, _ = replay_buffer.sample()

    with torch.no_grad():
        for key, val in q_var_dict.items():
            for i in range(args.num_aug):
                aug_obs = augmentations.aug_to_func[key](obs).to("cuda")
                Q1, Q2 = agent.critic(aug_obs, actions)
                critic_Q1, critic_Q2 = agent.critic_target(aug_obs, actions)

                if i == 0:
                    val["Q1"] = Q1
                    val["Q2"] = Q2
                    val["Q1-target"] = critic_Q1
                    val["Q2-target"] = critic_Q2
                else:
                    val["Q1"] = torch.cat((val["Q1"], Q1), axis=1)
                    val["Q2"] = torch.cat((val["Q2"], Q2), axis=1)
                    val["Q1-target"] = torch.cat((val["Q1-target"], critic_Q1), axis=1)
                    val["Q2-target"] = torch.cat((val["Q2-target"], critic_Q2), axis=1)

    data = list()
    for key, val in q_var_dict.items():
        row = [
            key,
            val["Q1"].std(dim=1).mean().item(),
            val["Q2"].std(dim=1).mean().item(),
            val["Q1-target"].std(dim=1).mean().item(),
            val["Q2-target"].std(dim=1).mean().item(),
        ]
        data.append(row)

    pd.DataFrame(
        data, columns=["key", "Q1 std", "Q2 std", "Q1 target std", "Q2 target std"]
    ).to_csv("./augmentation_Q_variance.csv")


if __name__ == "__main__":
    args = parse_Q_variance_args()
    main(args)
