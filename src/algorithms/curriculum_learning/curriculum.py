from utils import ReplayBuffer
from algorithms.sac import SAC
import augmentations as augmentations


class Curriculum(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aug_func = augmentations.aug_to_func[args.data_aug]

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = self.aug_func(obs)

        self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)
