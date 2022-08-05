import torch
import gym
from algorithms.sac import SAC
import utils
import pandas as pd
from scipy.stats.stats import pearsonr
from arguments import add_SAC_args, format_args
from env.wrappers import make_env
from algorithms.factory import make_agent


def calculate_MC_returns(rewards, discount):
    mc_rewards = []
    total_discounted_reward = 0
    for i in range(len(rewards) - 1, -1, -1):
        reward = rewards[i].item()
        mc_reward = reward + discount * total_discounted_reward
        total_discounted_reward = mc_reward
        mc_rewards.insert(0, mc_reward)

    return mc_rewards


def calculate_Q_correlation_coeff(
    agent: SAC, replay_buffer: utils.ReplayBuffer, discount
):
    all_idxs = list(range(replay_buffer.idx))
    obs, actions, rewards, next_obs, not_dones = replay_buffer.sample(idxs=all_idxs)

    with torch.no_grad():
        target_Q = agent.calculate_target_Q(
            next_obs=next_obs, reward=rewards, not_done=not_dones
        )
        Q1, Q2 = agent.critic(obs, actions)

    episode_reward = torch.sum(rewards).item()
    mc_rewards = torch.FloatTensor(
        calculate_MC_returns(rewards=rewards, discount=discount)
    )

    return Q1, Q2, target_Q, mc_rewards


def roll_out_policy(agent: SAC, env):
    done = False
    obs = env.reset()
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=1000,
        batch_size=1,
    )
    while not done:
        action = agent.select_action(obs=obs)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(
            obs=obs, action=action, reward=reward, next_obs=next_obs, done=float(done)
        )
        obs = next_obs
        if replay_buffer.capacity <= replay_buffer.idx + 1:
            replay_buffer.capacity += 20
    assert (
        replay_buffer.not_dones[replay_buffer.idx - 1] == 0
    ), 'Last element in "not_dones" is not 0'
    return replay_buffer


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
        capacity=args.num_samples + 1,
        batch_size=args.batch_size,
    )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    print(f"Number of train samples {args.num_samples}")
    print(f"Number of train steps {args.train_steps}")
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    done = True
    for _ in range(args.num_samples):
        if done:
            obs = env.reset()
            done = False

        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs

    data = []

    for step in range(args.train_steps):
        agent.update(replay_buffer=replay_buffer, L=None, step=step)

        roll_out_replay_buffer = roll_out_policy(agent=agent, env=env)
        Q1, Q2, target_Q, mc_returns = calculate_Q_correlation_coeff(
            agent=agent, replay_buffer=roll_out_replay_buffer, discount=args.discount
        )

        Q1 = Q1.squeeze().detach().cpu().numpy()
        Q2 = Q2.squeeze().detach().cpu().numpy()
        target_Q = target_Q.squeeze().detach().cpu().numpy()
        mc_returns = mc_returns.detach().cpu().numpy()
        Q1_Q_target_cc = pearsonr(Q1, target_Q)[0]
        Q2_Q_target_cc = pearsonr(Q2, target_Q)[0]
        Q1_MC_return_cc = pearsonr(Q1, mc_returns)[0]
        Q2_MC_return_cc = pearsonr(Q2, mc_returns)[0]
        data.append(
            [step, Q1_Q_target_cc, Q2_Q_target_cc, Q1_MC_return_cc, Q2_MC_return_cc]
        )

    pd.DataFrame(
        data,
        columns=[
            "step",
            "Q1, Q-target CC",
            "Q2, Q-target CC",
            "Q1, MC Return CC",
            "Q2, MC Return CC",
        ],
    ).to_csv(f"{args.id}.csv", index=False)


if __name__ == "__main__":
    parser = add_SAC_args()
    parser.add_argument("--num_samples", default=15000, type=int)
    args = format_args(parser.parse_args())
    main(args)
