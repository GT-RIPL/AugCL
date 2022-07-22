import torch
import torch.nn.functional as F
from copy import deepcopy
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
        self.wsa_optimizer = torch.optim.Adam(
            self.wsa_encoder.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
        )
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, "predictor"):
            self.predictor.train(training)

    def compute_wsa_loss(self, x):
        loss = 0
        h = self.critic_target.encoder(x)

        for _ in range(3):
            x_strong = self.aug_func(x)
            h_strong = self.wsa_encoder(x_strong)
            loss += F.mse_loss(h, h_strong)
        return loss

    def select_action(self, obs):
        if self.training:
            mu = super().select_action(obs=obs)
        else:
            mu = super().select_action(obs=obs, aug_encoder=self.wsa_encoder)

        return mu

    def update_wsa(self, obs, L=None, step=None):
        wsa_loss = self.compute_wsa_loss(x=obs)

        self.wsa_optimizer.zero_grad()
        wsa_loss.backward()
        self.wsa_optimizer.step()

        if L is not None:
            L.log("train/wsa_loss", wsa_loss, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shifted = augmentations.random_shift(obs)

        self.update_critic(obs_shifted, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs=obs_shifted, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_wsa(obs=obs, L=L, step=step)
