from utils import ReplayBuffer
from algorithms.rad import RAD
from algorithms.sac import SAC
import utils


class Curriculum(RAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def load_pretrained_agent(self, pretrained_agent: SAC):
        utils.soft_update_params(
            net=pretrained_agent.actor, target_net=self.actor, tau=1
        )
        utils.soft_update_params(
            net=pretrained_agent.critic, target_net=self.critic, tau=1
        )
        utils.soft_update_params(
            net=pretrained_agent.critic_target, target_net=self.critic_target, tau=1
        )

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = self.apply_aug(obs)

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
