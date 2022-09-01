from utils import ReplayBuffer
from augmentations import random_shift
import torch.nn.functional as F
from algorithms.sac import SAC
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double
import utils


class Curriculum_Resample(Curriculum_Double):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def load_pretrained_agent(self, pretrained_agent: SAC):
        utils.soft_update_params(
            net=pretrained_agent.actor, target_net=self.actor, tau=1
        )
        utils.soft_update_params(
            net=pretrained_agent.critic, target_net=self.critic, tau=1
        )
        utils.soft_update_params(
            net=pretrained_agent.critic_target, target_net=self.critic_target, tau=1
        )
        self.critic_weak = pretrained_agent.critic
        self.critic_weak_optimizer = pretrained_agent.critic_optimizer

    def update_critic(
        self, obs_aug, obs_shift, action, reward, next_obs, not_done, L=None, step=None
    ):
        target_Q = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

        current_Q1_shift, current_Q2_shift = self.critic_weak(obs_shift, action)
        critic_loss_shift = F.mse_loss(current_Q1_shift, target_Q) + F.mse_loss(
            current_Q2_shift, target_Q
        )

        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action.repeat(2, 1, 1, 1))
        critic_loss_aug = F.mse_loss(
            current_Q1_aug, target_Q.repeat(2, 1, 1, 1)
        ) + F.mse_loss(current_Q2_aug, target_Q.repeat(2, 1, 1, 1))

        if L is not None:
            L.log("train_critic_weak/loss", critic_loss_shift, step)
            L.log("train_critic_strong/loss", critic_loss_aug, step)

        self.critic_weak_optimizer.zero_grad()
        critic_loss_shift.backward()
        self.critic_weak_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss_aug.backward()
        self.critic_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(
            self.critic_weak.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic_weak.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic_weak.encoder, self.critic_target.encoder, self.encoder_tau
        )

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = random_shift(obs)
        obs_aug = self.apply_aug(obs.repeat(2, 1, 1, 1))

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
            self.update_actor_and_alpha(obs_aug, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
