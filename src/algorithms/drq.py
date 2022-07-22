from algorithms.sac import SAC
from utils import ReplayBuffer


class DrQ(SAC):  # [K=1, M=1]
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.k = args.drq_k
        self.m = args.drq_m

    def update(self, replay_buffer: ReplayBuffer, L, step):
        if self.k == 1 and self.m == 1:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()
            self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        else:
            (
                obs_list,
                action,
                reward,
                next_obs_list,
                not_done,
            ) = replay_buffer.sample_drq_with_k_and_m(k=self.k, n=self.n)
            raise NotImplemented("Multi k and multi m not implemented yet for DrQ")

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
