import torch
from copy import deepcopy
import utils
from utils import ReplayBuffer
from augmentations import random_shift
import torch.nn.functional as F
from algorithms.sac import SAC
from algorithms.curriculum_learning.curriculum import Curriculum


class Curriculum_FTL(Curriculum):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.actor_aug = deepcopy(self.actor)
        self.actor_aug_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.mse_loss = torch.nn.MSELoss()

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            if self.training:
                mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
            else:
                mu, _, _, _ = self.actor_aug(
                    _obs, compute_pi=False, compute_log_pi=False
                )

        return mu.cpu().data.numpy().flatten()

    def load_pretrained_agent(self, pretrained_agent: SAC):
        self.actor = pretrained_agent.actor
        self.critic = pretrained_agent.critic
        self.critic_target = pretrained_agent.critic_target
        self.actor_optimizer = pretrained_agent.actor_optimizer
        self.critic_optimizer = pretrained_agent.critic_optimizer
        self.log_alpha = pretrained_agent.log_alpha
        utils.soft_update_params(net=self.actor, target_net=self.actor_aug, tau=1)

    def calculate_target_Q(self, next_obs, reward, not_done):
        with torch.no_grad():
            mu, policy_action, log_pi, log_std = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        return target_Q, mu, log_std

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        target_Q, mu, log_std = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return mu, log_std

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        mu, pi, log_pi, log_std = self.actor(obs, detach=True)
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

        return mu, log_std

    def update_augmented_actor(self, obs, obs_mu, obs_log_std):
        mu, _, _, log_std = self.actor_aug(obs)

        MSE_mu = self.mse_loss(mu, obs_mu)
        MSE_std = self.mse_loss(obs_log_std, log_std)

        self.actor_aug_optimizer.zero_grad()
        (MSE_mu + MSE_std).backward()
        self.actor_aug_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = random_shift(obs)

        next_obs_mu, next_obs_log_std = self.update_critic(
            obs=obs_shift,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            obs_mu, obs_log_std = self.update_actor_and_alpha(obs_shift, L, step)
        else:
            obs_mu, _, _, obs_log_std = self.actor(obs_shift)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        obs_all = torch.cat((obs, next_obs), dim=0)
        mu_all = torch.cat((obs_mu, next_obs_mu), dim=0)
        std_all = torch.cat((obs_log_std, next_obs_log_std), dim=0)
        obs_all_aug = self.apply_aug(obs_all)
        self.update_augmented_actor(obs=obs_all_aug, obs_mu=mu_all, obs_log_std=std_all)
