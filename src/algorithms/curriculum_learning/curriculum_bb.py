from augmentations import random_shift
from algorithms.curriculum_learning.curriculum_2x_nu_actor import Curriculum_2x_Nu_Actor
from utils import ReplayBuffer


class Curriculum_BB(Curriculum_2x_Nu_Actor):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

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
            self.update_actor_and_alpha(obs_shift=obs, obs_aug=obs_aug, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
