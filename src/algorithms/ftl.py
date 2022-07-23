import torch
import utils
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import algorithms.modules as m
from algorithms.sac import SAC
import augmentations


class FTL(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = (
            augmentations.random_overlay
            if args.use_overlay
            else augmentations.random_conv
        )

        self.FTL_critic = deepcopy(self.critic)
        self.FTL_actor = deepcopy(self.actor)
        self.FTL_log_alpha = deepcopy(self.log_alpha)

        self.FTL_actor_optimizer = torch.optim.Adam(
            self.FTL_actor.parameters(),
            lr=args.actor_lr,
            betas=(args.actor_beta, 0.999),
        )
        self.FTL_critic_optimizer = torch.optim.Adam(
            self.FTL_critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
        )
        self.FTL_log_alpha_optimizer = torch.optim.Adam(
            [self.FTL_log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, "FTL_actor"):
            self.FTL_actor.train(training)
            self.FTL_critic.train(training)

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            if self.training:
                mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
            else:
                mu, _, _, _ = self.FTL_actor(
                    _obs, compute_pi=False, compute_log_pi=False
                )
        return mu.cpu().data.numpy().flatten()

    @property
    def FTL_alpha(self):
        return self.FTL_log_alpha.exp()

    def update_critic(
        self, obs, aug_obs, action, reward, next_obs, not_done, L=None, step=None
    ):
        with torch.no_grad():
            # Weak Aug
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            # Strong Aug
            aug_next_obs = self.aug_func(next_obs)
            _, FTL_policy_action, FTL_log_pi, _ = self.FTL_actor(aug_next_obs)
            FTL_target_Q1, FTL_target_Q2 = self.critic_target(
                next_obs, FTL_policy_action
            )
            FTL_target_V = (
                torch.min(FTL_target_Q1, FTL_target_Q2)
                - self.alpha.detach() * FTL_log_pi
            )
            FTL_target_Q = reward + (not_done * self.discount * FTL_target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        FTL_current_Q1, FTL_current_Q2 = self.FTL_critic(aug_obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        FTL_critic_loss = F.mse_loss(FTL_current_Q1, FTL_target_Q) + F.mse_loss(
            FTL_current_Q2, target_Q
        )

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)
            L.log("train_FTL_critic/loss", FTL_critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.FTL_critic_optimizer.zero_grad()
        FTL_critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(
        self, obs, aug_obs, L=None, step=None, update_alpha=True
    ):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        _, FTL_pi, FTL_log_pi, FTL_log_std = self.FTL_actor(aug_obs, detach=True)
        FTL_actor_Q1, FTL_actor_Q2 = self.FTL_critic(aug_obs, FTL_pi, detach=True)

        FTL_actor_Q = torch.min(FTL_actor_Q1, FTL_actor_Q2)
        FTL_actor_loss = (self.FTL_alpha.detach() * FTL_log_pi - FTL_actor_Q).mean()

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )
            L.log("train_actor/loss", actor_loss, step)
            L.log("train_actor/mean_entropy", entropy.mean(), step)

            FTL_entropy = 0.5 * FTL_log_std.shape[1] * (
                1.0 + np.log(2 * np.pi)
            ) + FTL_log_std.sum(dim=-1)
            L.log("train_FTL_actor/loss", FTL_actor_loss, step)
            L.log("train_FTL_actor/mean_entropy", FTL_entropy.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.FTL_actor_optimizer.zero_grad()
        FTL_actor_loss.backward()
        self.FTL_actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.FTL_log_alpha_optimizer.zero_grad()
            FTL_alpha_loss = (
                self.FTL_alpha * (-FTL_log_pi - self.target_entropy).detach()
            ).mean()

            if L is not None:
                L.log("train_FTL_alpha/loss", FTL_alpha_loss, step)
                L.log("train_FTL_alpha/value", self.alpha, step)

            FTL_alpha_loss.backward()
            self.FTL_log_alpha_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
        utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder, self.encoder_tau
        )

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = augmentations.random_shift(imgs=obs)
        aug_obs = self.aug_func(obs)

        self.update_critic(obs, aug_obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, aug_obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
