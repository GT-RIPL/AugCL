from copy import deepcopy
from augmentations import random_shift
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double
from algorithms.sac import SAC
from utils import ReplayBuffer


class Curriculum_2x_Opt(Curriculum_Double):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def load_pretrained_agent(self, pretrained_agent: SAC):
        super().load_pretrained_agent(pretrained_agent=pretrained_agent)
        self.actor = deepcopy(pretrained_agent.actor)
        self.actor_optimizer = deepcopy(pretrained_agent.actor_optimizer)

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
            self.update_actor_and_alpha(obs=obs, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
