import json
import augmentations
from algorithms.sac import SAC


class RAD(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        aug_keys = args.data_aug.split("-")
        aug_params = json.loads(args.aug_params) if args.aug_params else {}
        self.aug_funcs = dict()

        for key in aug_keys:
            self.aug_funcs[key] = dict(
                func=augmentations.aug_to_func[key], params=aug_params.get(key, {})
            )

    def apply_aug(self, x):
        for _, aug_dict in self.aug_funcs.items():
            x = aug_dict["func"](x, **aug_dict["params"])

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
