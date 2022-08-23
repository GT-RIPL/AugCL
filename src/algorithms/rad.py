import augmentations
from algorithms.sac import SAC


class RAD(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_keys = args.data_aug.split("-")
        self.aug_funcs = [augmentations.aug_to_func[key] for key in self.aug_keys]

    def apply_aug(self, x):
        for func in self.aug_funcs:
            x = func(x)

        return x

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        obs = self.apply_aug(obs)
        next_obs = self.apply_aug(next_obs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
