import torch
import numpy as np
from augmentations import random_shift
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double
from utils import ReplayBuffer


class Curriculum_2x_Nu_Actor(Curriculum_Double):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update_actor_and_alpha(
        self, obs_aug, obs_shift, L=None, step=None, update_alpha=True
    ):
        _, pi, log_pi, log_std = self.actor(obs_aug, detach=True)
        actor_Q1, actor_Q2 = self.critic_weak(obs_shift, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )
            L.log("train_actor/loss", actor_loss, step)
            L.log("train_actor/mean_entropy", entropy.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = random_shift(obs)
        obs_aug = self.apply_aug(obs)

        self.update_critic(
            obs_shift=obs_shift,
            obs_aug=obs_aug,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(
                obs_shift=obs_shift, obs_aug=obs_aug, L=L, step=step
            )

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
