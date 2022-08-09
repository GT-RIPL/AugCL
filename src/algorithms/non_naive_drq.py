import augmentations
from algorithms.drq import DrQ
from utils import ReplayBuffer


class NonNaiveDrQ(DrQ):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_list = [augmentations.random_shift(obs) for _ in range(self.k)]
        next_obs_list = [next_obs]

        self.update_critic(
            obs_list=obs_list,
            action=action,
            reward=reward,
            next_obs_list=next_obs_list,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_list[0], L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
