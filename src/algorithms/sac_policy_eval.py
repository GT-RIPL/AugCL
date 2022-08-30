import torch
from algorithms.rad import RAD
import torch.nn.functional as F


class SAC_policy_eval(RAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def calculate_target_Q(self, next_obs, reward, not_done):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, _ = self.critic_target(next_obs, policy_action)
            target_V = target_Q1 - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        return target_Q

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        target_Q = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

        current_Q1, _ = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q)
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = self.apply_aug(obs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
