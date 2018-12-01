import sys
import random
import logging
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from visual_nav.utils.replay_buffer import ReplayBuffer


class ReinforcementLearner(object):
    def __init__(self,
                 env,
                 q_func,
                 device,
                 output_dir,
                 replay_buffer_size=1000000,
                 batch_size=128,
                 gamma=0.9,
                 frame_history_len=4,
                 target_update_freq=10000,
                 num_test_case=100,
                 use_best_wts=False,
                 regression=False
                 ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.output_dir = output_dir
        self.num_test_case = num_test_case
        self.use_best_wts = use_best_wts

        img_h, img_w, img_c = env.observation_space.shape
        self.input_arg = frame_history_len * img_c
        self.num_actions = env.action_space.n
        self.image_size = (img_h, img_w, img_c)
        self.time_step = env.unwrapped.time_step

        self.Q = q_func(self.input_arg, self.num_actions, regression=regression).to(device)
        self.target_Q = q_func(self.input_arg, self.num_actions, regression=regression).to(device)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, self.image_size)
        self.replay_buffer = None

        self.log_every_n_steps = 10000
        self.num_param_updates = 0

    def test(self):
        logging.info('Start testing model')
        replay_buffer = ReplayBuffer(int(self.num_test_case * self.env.max_time / self.env.time_step),
                                     self.frame_history_len, self.image_size)

        for i in range(self.num_test_case):
            obs = self.env.reset()
            done = False
            while not done:
                last_idx = replay_buffer.store_observation(obs)
                recent_observations = replay_buffer.encode_recent_observation()
                action = self.act(recent_observations)
                obs, reward, done, info = self.env.step(action.item())
                replay_buffer.store_effect(last_idx, action, reward, done)

        logging.info(self.env.get_episodes_summary(num_last_episodes=self.num_test_case))

    def train(self, optimizer_spec, exploration, learning_starts=50000,
              learning_freq=4, num_timesteps=2000000, episode_update=False):
        statistics_file = os.path.join(self.output_dir, 'statistics.json')
        weights_file = os.path.join(self.output_dir, 'rl_model.pth')
        logging.info('Start reinforcement learning')
        writer = SummaryWriter()
        episode_starts = len(self.env.get_episode_rewards())
        avg_reward = -float('nan')
        success_rate = -float('nan')
        collision_rate = -float('nan')
        overtime_rate = -float('nan')
        avg_time = -float('nan')
        best_avg_episode_reward = -float('inf')
        last_obs = self.env.reset()
        optimizer = optimizer_spec.constructor(self.Q.parameters(), **optimizer_spec.kwargs)

        t = 0
        while True:
            # Check stopping criterion
            if self.env.get_total_steps() > num_timesteps:
                break

            if not episode_update:
                last_idx = self.replay_buffer.store_observation(last_obs)
                recent_observations = self.replay_buffer.encode_recent_observation()

                # Choose random action if not yet start learning
                if t > learning_starts:
                    eps_threshold = exploration.value(t)
                    action = self._select_epsilon_greedy_action(self.Q, recent_observations, eps_threshold)[0]
                else:
                    action = torch.IntTensor([[random.randrange(self.num_actions)]])
                # Advance one step
                obs, reward, done, info = self.env.step(action.item())
                # Store other info in replay memory
                self.replay_buffer.store_effect(last_idx, action, reward, done)
                # Resets the environment when reaching an episode boundary.
                if done:
                    logging.info(self.env.get_episode_summary() + ' in step {}'.format(t))
                    obs = self.env.reset()
                last_obs = obs

                if (t > learning_starts and t % learning_freq == 0 and
                        self.replay_buffer.can_sample(self.batch_size)):
                    self._td_update(optimizer)
                t += 1
            else:
                done = False
                observations = []
                frames = []
                goals = []
                effects = []
                while not done:
                    frame, goal = last_obs
                    frame = np.array(frame).astype(np.float32)
                    goal = np.array(goal).astype(np.float32)

                    # transpose image frame into (img_c, img_h, img_w)
                    frame = frame.transpose(2, 0, 1)
                    frames.append(frame)
                    goals.append(goal)

                    frame_concat = []
                    goal_concat = []
                    if len(frames) < self.frame_history_len:
                        for _ in range(self.frame_history_len - len(frames)):
                            frame_concat.append(np.zeros_like(frame))
                            goal_concat.append(np.zeros_like(goal))
                        frame_concat += frames
                        goal_concat += goals
                    else:
                        frame_concat = frames[-4:]
                        goal_concat = goals[-4:]
                    frame_concat = np.concatenate(frame_concat, 0)
                    goal_concat = np.concatenate(goal_concat, 0)
                    recent_observations = frame_concat, goal_concat

                    if t > learning_starts:
                        eps_threshold = exploration.value(t)
                        action = self._select_epsilon_greedy_action(self.Q, recent_observations, eps_threshold)[0]
                    else:
                        action = torch.IntTensor([[random.randrange(self.num_actions)]])

                    obs, reward, done, info = self.env.step(action.item())

                    observations.append(last_obs)
                    effects.append((action, reward, done))
                    last_obs = obs
                    t += 1

                if info in ['Success', 'Collision']:
                    # only update the replay buffer if the robot has positive reward
                    for obs, effect in zip(observations, effects):
                        last_idx = self.replay_buffer.store_observation(obs)
                        self.replay_buffer.store_effect(last_idx, *effect)
                        if (t > learning_starts and t % learning_freq == 0 and
                                self.replay_buffer.can_sample(self.batch_size)):
                            self._td_update(optimizer)

                last_obs = self.env.reset()
                logging.info(self.env.get_episode_summary() + ' in step {}'.format(t))

            # Log progress and keep track of statistics
            num_last_episodes = 100
            episode_rewards = self.env.get_episode_rewards()[episode_starts:]
            num_episodes = len(episode_rewards)
            if num_episodes > 0:
                avg_reward = self.env.get_average_reward(num_last_episodes, episode_starts)
                success_rate = self.env.get_success_rate(num_last_episodes, episode_starts)
                collision_rate = self.env.get_collision_rate(num_last_episodes, episode_starts)
                overtime_rate = self.env.get_overtime_rate(num_last_episodes, episode_starts)
                avg_time = self.env.get_average_time(num_last_episodes, episode_starts)
            if num_episodes > num_last_episodes:
                best_avg_episode_reward = max(best_avg_episode_reward, avg_reward)

            writer.add_scalar('data/mean_episode_rewards', avg_reward, t)
            writer.add_scalar('data/best_mean_episode_rewards', best_avg_episode_reward, t)
            writer.add_scalar('data/success_rate', success_rate, t)
            writer.add_scalar('data/collision_rate', collision_rate, t)
            writer.add_scalar('data/overtime_rate', overtime_rate, t)
            writer.add_scalar('data/mean_episode_time', avg_time, t)

            if t % self.log_every_n_steps == 0 and t > learning_starts:
                logging.info("Timestep %d" % (t,))
                logging.info("mean reward (100 episodes) %f" % avg_reward)
                logging.info("best mean reward %f" % best_avg_episode_reward)
                logging.info("episodes %d" % num_episodes)
                logging.info("exploration %f" % exploration.value(t))
                sys.stdout.flush()

                # Dump statistics to json file
                writer.export_scalars_to_json(statistics_file)
                logging.info("Saved to %s" % statistics_file)

                torch.save(self.Q.state_dict(), weights_file)

        writer.close()

    def _select_epsilon_greedy_action(self, model, obs, eps_threshold):
        sample = random.random()
        if sample > eps_threshold:
            frames = torch.from_numpy(obs[0]).unsqueeze(0).to(self.device) / 255.0
            goals = torch.from_numpy(obs[1]).unsqueeze(0).to(self.device)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return model(frames, goals).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(self.num_actions)])

    def act(self, obs, model=None):
        frames = torch.from_numpy(obs[0]).unsqueeze(0).to(self.device) / 255.0
        goals = torch.from_numpy(np.array(obs[1])).unsqueeze(0).to(self.device)
        if model:
            return model(frames, goals).data.max(1)[1].cpu()
        else:
            return self.Q(frames, goals).data.max(1)[1].cpu()

    def _td_update(self, optimizer):
        # Use the replay buffer to sample a batch of transitions
        # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target
        frames_batch, goals_batch, action_batch, reward_batch, next_frames_batch, next_goals_batch, done_mask = \
            self.replay_buffer.sample(self.batch_size)
        # Convert numpy nd_array to torch variables for calculation
        frames_batch = torch.from_numpy(frames_batch).to(self.device) / 255.0
        goals_batch = torch.from_numpy(goals_batch).to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).to(self.device)
        next_frames_batch = torch.from_numpy(next_frames_batch).to(self.device) / 255.0
        next_goals_batch = torch.from_numpy(next_goals_batch).to(self.device)
        not_done_mask = torch.from_numpy(1 - done_mask).to(self.device)

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken, action is used to index the value in the dqn output
        # current_q_values[i][j] = Q_outputs[i][action_batch[i][j]], where j=0
        current_q_values = self.Q(frames_batch, goals_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_Q(next_frames_batch, next_goals_batch).detach().max(1)[0]
        next_q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_q_values = reward_batch + (pow(self.gamma, self.time_step) * next_q_values)

        # Compute Bellman error
        td_error = target_q_values - current_q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = td_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient w.r.t current_q_values
        # Cuz in the td_error, there is a negative sing before current_q_values
        d_error = clipped_bellman_error * -1.0
        # Clear previous gradients before backward pass
        optimizer.zero_grad()
        # run backward pass and back prop through Q network, d_error is the gradient of final loss w.r.t. Q
        current_q_values.backward(d_error.data)

        # Perform the update
        optimizer.step()
        self.num_param_updates += 1

        # Periodically update the target network by Q network to target Q network
        if self.num_param_updates % self.target_update_freq == 0:
            self.target_Q.load_state_dict(self.Q.state_dict())

    def _mc_update(self, optimizer, criterion, num_train_batch=1):
        for _ in range(num_train_batch):
            frames_batch, goals_batch, action_batch, _, _, _, _, value_batch = \
                self.replay_buffer.sample(self.batch_size, with_value=True)
            # Convert numpy nd_array to torch variables for calculation
            frames_batch = torch.from_numpy(frames_batch).to(self.device) / 255.0
            goals_batch = torch.from_numpy(goals_batch).to(self.device)
            action_batch = torch.from_numpy(action_batch).long().to(self.device)
            value_batch = torch.from_numpy(value_batch).to(self.device)

            current_q_values = self.Q(frames_batch, goals_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            loss = criterion(current_q_values, value_batch)
            optimizer.zero_grad()
            loss.backward()
            logging.info('Batch loss: {:.4f}'.format(loss.item()))

            # Perform the update
            optimizer.step()
            self.num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if self.num_param_updates % self.target_update_freq == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())
