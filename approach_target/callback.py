import numpy as np
from os.path import join as PJ
from stable_baselines3.common.callbacks import BaseCallback


class TrainLogCallback(BaseCallback):
    """ Save and display training process. """

    def __init__(self, freq, result_dir, num_timesteps, verbose=1):
        super().__init__(verbose)

        self.freq = freq
        self.log_dir = PJ(result_dir, 'logs')
        self.reward_history = []
        self.num_timesteps = num_timesteps

    def _on_step(self) -> bool:
        """ This method will be called by the model after each call to
            `env.step()`.
        """
        if (self.num_timesteps + 1) % self.freq != 0:
            return True

        info_dict = self.locals['infos'][0]
        cur_reward = info_dict['reward']
        num_episode = info_dict['num_episode']

        self.logger.record_mean('Train/reward', cur_reward)
        self.logger.record('Train/num_achieve_target',
                           info_dict['num_complete'] / num_episode)
        self.logger.record('Train/num_broken',
                           info_dict['num_broken'] / num_episode)

        # Mean reward
        self.reward_history.append(cur_reward)
        self.reward_history = self.reward_history[-1000:]
        mean_reward = np.mean(self.reward_history)

        print(f"[{str(self.num_timesteps):6s}] Reward: {cur_reward:.4f} | Mean Reward: {mean_reward:.4f}")
        print(f"         num_complete: {info_dict['num_complete']}")
        print(f"         num_broken: {info_dict['num_broken']}")
        print(f"         num_episode: {info_dict['num_episode']}")

        # Write into tenorboard directly
        self.logger.dump(self.num_timesteps)
        return True
