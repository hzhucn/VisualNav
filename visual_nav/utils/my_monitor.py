import numpy as np
from gym.wrappers.monitor import Monitor


class MyMonitor(Monitor):
    def __init__(self, env, directory):
        super().__init__(env, directory, resume=True)
        self.time_step = env.time_step
        self.max_time = env.max_time
        self.successes = list()
        self.collisions = list()
        self.overtimes = list()
        self.last_done_info = ''

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)

        if done:
            self.successes.append(1 if info == 'Success' else 0)
            self.collisions.append(1 if info == 'Collision' else 0)
            self.overtimes.append(1 if info == 'Overtime' else 0)
            self.last_done_info = info

        return observation, reward, done, info

    def get_success_rate(self, num_last_episodes, episode_starts=0):
        return np.mean(self.successes[episode_starts:][-num_last_episodes:])

    def get_collision_rate(self, num_last_episodes, episode_starts=0):
        return np.mean(self.collisions[episode_starts:][-num_last_episodes:])

    def get_overtime_rate(self, num_last_episodes, episode_starts=0):
        return np.mean(self.overtimes[episode_starts:][-num_last_episodes:])

    def get_average_time(self, num_last_episodes, episode_starts=0):
        return np.mean(self.stats_recorder.episode_lengths[episode_starts:][-num_last_episodes:]) * self.time_step

    def get_average_reward(self, num_last_episodes, episode_starts=0):
        return np.mean(self.stats_recorder.episode_rewards[episode_starts:][-num_last_episodes:])

    def get_episodes_summary(self, num_last_episodes):
        success_rate = self.get_success_rate(num_last_episodes)
        collision_rate = self.get_collision_rate(num_last_episodes)
        overtime_rates = self.get_overtime_rate(num_last_episodes)
        avg_time = self.get_average_time(num_last_episodes)
        avg_reward = self.get_average_reward(num_last_episodes)

        return 'Success: {:.2f}, collision: {:.2f}, overtime: {:.2f}, avg steps: {:.2f}s, avg reward: {:.4f}'.\
            format(success_rate, collision_rate, overtime_rates, avg_time, avg_reward)

    def get_episode_summary(self, ):
        return 'Episode finished in {:<5}s with total reward {:<5.2f} and end signal {:<10}'.\
                 format(self.stats_recorder.episode_lengths[-1] * self.time_step,
                        self.stats_recorder.episode_rewards[-1], self.last_done_info)

