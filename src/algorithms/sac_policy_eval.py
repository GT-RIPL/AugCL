from algorithms.sac import SAC


class SAC_policy_eval(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
