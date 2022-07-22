import torch
import torch.nn.functional as F
from algorithms.sac import SAC
from utils import ReplayBuffer
import augmentations


class DArQ(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.k = args.drq_k
        self.m = args.drq_m
        self.num_train_steps = args.train_steps

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            next_obs = augmentations.random_shift(next_obs)
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        critic_loss = 0

        for _ in range(self.k):
            p = (self.num_train_steps - step) / (2 * self.num_train_steps)
            x_encoded = F.dropout(
                self.critic.encoder(augmentations.random_shift(obs)), p=p
            )
            current_Q1 = self.critic.Q1(x_encoded, action)
            current_Q2 = self.critic.Q2(x_encoded, action)
            critic_loss += F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
