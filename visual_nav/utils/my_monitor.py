import logging
from gym.wrappers.monitor import Monitor


class MyMonitor(Monitor):
    def __init__(self, env, directory):
        super().__init__(env, directory)
        self.time_step = env.time_step
        self.successes = list()
        self.collisions = list()
        self.overtimes = list()
        self.last_done_info = ''

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)

        if done:
            self.successes.append(1 if info == 'Accomplishment' else 0)
            self.collisions.append(1 if info == 'Collision' else 0)
            self.overtimes.append(1 if info == 'Overtime' else 0)
            self.last_done_info = info

        return observation, reward, done, info

    def print_episodes_summary(self, num_last_episodes):
        success_rate = sum(self.successes[-num_last_episodes:]) / num_last_episodes
        collision_rate = sum(self.collisions[-num_last_episodes:]) / num_last_episodes
        overtime_rates = sum(self.overtimes[-num_last_episodes:]) / num_last_episodes
        avg_time = sum(self.stats_recorder.episode_lengths[-num_last_episodes:]) / num_last_episodes * self.time_step
        avg_rewards = sum(self.stats_recorder.episode_rewards[-num_last_episodes:]) / num_last_episodes

        logging.info('Success: {:.2f}, collision: {:.2f}, overtime: {:.2f}, avg steps: {:.2f}s, avg reward: {:.4f}'.
                     format(success_rate, collision_rate, overtime_rates, avg_time, avg_rewards))

    def print_episode_summary(self):
        logging.info('Episode finished in {}s with total reward {:.4f} and end signal {}'.
                     format(self.stats_recorder.episode_lengths[-1] * self.time_step,
                            self.stats_recorder.episode_rewards[-1], self.last_done_info))

