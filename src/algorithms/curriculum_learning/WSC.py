import torch
from utils import ReplayBuffer
import torch.nn.functional as F
from algorithms.sac import SAC
import augmentations as augmentations
import algorithms.modules as m


class WSC(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = augmentations.aug_to_func[args.data_aug]

        shared_cnn = m.SharedCNN(
            obs_shape, args.num_shared_layers, args.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, args.num_head_layers, args.num_filters
        ).cuda()
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        self.strong_critic = m.Critic(
            critic_encoder, action_shape, args.hidden_dim
        ).cuda()
        self.strong_critic_optimizer = torch.optim.Adam(
            self.strong_critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
        )

        self.actor.encoder = self.strong_critic.encoder

    def train(self, training=True):
        super().train(training=training)
        if hasattr(self, "strong_critic"):
            self.strong_critic.train(training)

    def eval(self):
        self.train(False)

    def update_critic(
        self, obs, obs_aug, action, reward, next_obs, not_done, L=None, step=None
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        current_Q1, current_Q2 = self.strong_critic(obs_aug, action)
        strong_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)
            L.log("train_critic/strong_loss", strong_critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.strong_critic_optimizer.zero_grad()
        strong_critic_loss.backward()
        self.strong_critic_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        aug_obs = self.aug_func(obs)
        obs = augmentations.random_shift(obs)

        self.update_critic(
            obs=obs,
            obs_aug=aug_obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        # if step % self.critic_target_update_freq == 0:
        #     self.soft_update_critic_target()
