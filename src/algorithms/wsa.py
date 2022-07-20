import copy
import torch
import torch.nn.functional as F
from logger import Logger
import utils
import augmentations
from algorithms.sac import SAC
import algorithms.modules as m


class WSA(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.augmented_encoder = copy.deepcopy(self.critic.encoder).cuda()
        self.augmented_critic = m.Critic(
            self.augmented_encoder, action_shape, args.hidden_dim
        ).cuda()
        self.augmented_critic_opt = torch.optim.Adam(
            self.augmented_critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
        )
        self.train()

        self.aug_func = (
            augmentations.random_overlay
            if args.use_overlay
            else augmentations.random_conv
        )

    def train(self, training=True):
        super().train(training)
        if hasattr(self, "augmented_encoder"):
            self.augmented_encoder.train(mode=training)

    def eval(self):
        super().train(False)
        self.augmented_encoder.train(mode=False)

    def select_action(self, obs):
        if self.training:
            mu = super().select_action(obs=obs)
        else:
            mu = super().select_action(obs=obs, aug_encoder=self.augmented_encoder)

        return mu

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        obs_aug = self.aug_func(obs.clone())
        current_Q1_aug, current_Q2_aug = self.augmented_critic(obs_aug, action)
        critic_loss_aug = F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(
            current_Q2_aug, target_Q
        )

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)
            L.log("train_critic/loss_aug", critic_loss_aug, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        torch.autograd.set_detect_anomaly(True)

        self.actor_optimizer.zero_grad()
        critic_loss_aug.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer: utils.ReplayBuffer, L: Logger, step: int):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
