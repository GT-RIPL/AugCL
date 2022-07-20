import torch
import utils
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import algorithms.modules as m
from algorithms.sac import SAC
import augmentations


class WSA(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aux_update_freq = args.aux_update_freq
        self.soda_batch_size = args.soda_batch_size
        self.soda_tau = args.soda_tau
        self.aug_func = (
            augmentations.random_overlay
            if args.use_overlay
            else augmentations.random_conv
        )

        self.wsa_encoder = deepcopy(self.critic.encoder)

        self.predictor = m.WSAPredictor(
            encoder=self.wsa_encoder,
            projection=m.SODAMLP(
                self.wsa_encoder.out_dim, args.projection_dim, self.wsa_encoder.out_dim
            ),
            hidden_dim=self.wsa_encoder.out_dim,
        ).cuda()
        self.predictor_target = m.SODAPredictor(
            self.critic.encoder, args.projection_dim
        ).cuda()

        self.soda_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
        )
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, "predictor"):
            self.predictor.train(training)

    def compute_soda_loss(self, x0, x1):
        h0 = self.predictor(x0)
        with torch.no_grad():
            h1 = self.predictor_target(x1)
        h0 = F.normalize(h0, p=2, dim=1)
        h1 = F.normalize(h1, p=2, dim=1)

        return F.mse_loss(h0, h1)

    def select_action(self, obs):
        if self.training:
            mu = super().select_action(obs=obs)
        else:
            mu = super().select_action(obs=obs, aug_encoder=self.wsa_encoder)

        return mu

    def update_wsa(self, obs, aug_obs, L=None, step=None):
        soda_loss = self.compute_soda_loss(aug_obs, obs)

        self.soda_optimizer.zero_grad()
        soda_loss.backward()
        self.soda_optimizer.step()

        if L is not None:
            L.log("train/aux_loss", soda_loss, step)

        utils.soft_update_params(
            self.predictor.projection, self.predictor_target.mlp, self.soda_tau
        )

    def update_actor_and_alpha(
        self, obs, aug_obs, L=None, step=None, update_alpha=True
    ):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)
        _, _, log_pi_aug, _ = self.actor(
            x=aug_obs, aug_encoder=self.predictor.encoder, detach=True
        )

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        actor_loss += (self.alpha.detach() * log_pi_aug - actor_Q).mean()
        actor_loss /= 2

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
            alpha_loss += (self.alpha * (-log_pi_aug - self.target_entropy).detach()).mean()
            alpha_loss /= 2

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shifted = augmentations.random_shift(obs)
        obs_aug = self.aug_func(obs.clone())

        self.update_critic(obs_shifted, action, reward, next_obs, not_done, L, step)

        if step % self.aux_update_freq == 0:
            self.update_wsa(obs=obs, aug_obs=obs_aug, L=L, step=step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs=obs, aug_obs=obs_aug, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
