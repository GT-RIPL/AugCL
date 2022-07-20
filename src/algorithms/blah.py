import torch
import numpy as np
from logger import Logger
import utils
import augmentations
from algorithms.sac_no_share import SAC_no_share


class BLAH(SAC_no_share):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = (
            augmentations.random_overlay
            if args.use_overlay
            else augmentations.random_conv
        )

    def update_actor_and_alpha(
        self, obs, aug_obs, L=None, step=None, update_alpha=True
    ):
        _, pi, log_pi, log_std = self.actor(aug_obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

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

    def update(self, replay_buffer: utils.ReplayBuffer, L: Logger, step: int):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = augmentations.random_shift(obs)
        obs_aug = self.aug_func(obs)

        self.update_critic(obs_shift, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs=obs, aug_obs=obs_aug, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def select_action(self, obs, aug_encoder=None):
        _obs = self._obs_to_input(obs)
        _obs = self.aug_func(_obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False, aug_encoder=aug_encoder
            )
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        _obs = self.aug_func(_obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()
