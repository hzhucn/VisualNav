import sys
import time
import copy
import random
import logging
import os
import argparse
import shutil
import pprint
import configparser
import importlib.util
from collections import namedtuple
from operator import itemgetter
from collections import defaultdict

import git
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.sarl import SARL
from visual_sim.envs.visual_sim import VisualSim
from visual_nav.utils.replay_buffer import ReplayBuffer, BufferWrapper, pack_batch
from visual_nav.utils.my_monitor import MyMonitor
from visual_nav.utils.schedule import LinearSchedule, ConstantSchedule
from visual_nav.utils.visualization_tools import heatmap, top_down_view
from visual_nav.utils.models import model_factory


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class Trainer(object):
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
        self.regression = regression

        img_h, img_w, img_c = env.observation_space.shape
        self.input_arg = frame_history_len * img_c
        self.num_actions = env.action_space.n
        self.image_size = (img_h, img_w, img_c)
        self.time_step = env.unwrapped.time_step

        self.Q = q_func(self.input_arg, self.num_actions, regression=regression).to(device)
        self.target_Q = q_func(self.input_arg, self.num_actions, regression=regression).to(device)
        # self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, self.image_size)
        self.replay_buffer = None

        self.log_every_n_steps = 10000
        self.num_param_updates = 0
        # map action_rot to its index and action_xy
        self.action_dict = {action: (i, ActionXY(action.v * np.cos(action.r), action.v * np.sin(action.r)))
                            for i, action in enumerate(self.env.unwrapped.actions)}
        self.idx2action = torch.from_numpy(np.stack([action for action in self.env.unwrapped.actions])).float()

    def imitation_learning(self, num_episodes=3000, training='mc', num_epochs=500, step_size=100, test=False):
        """
        Imitation learning and reinforcement learning share the same environment, replay buffer and Q function

        """
        il_max_time = 20
        num_train_batch = num_episodes * 50
        optimizer = optim.Adam(self.Q.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(int(num_episodes * il_max_time / self.env.time_step),
                                          self.frame_history_len, self.image_size)

        logging.info('Start imitation learning')
        weights_file = os.path.join(self.output_dir, 'il_model.pth')
        replay_buffer_file = 'data/replay_buffer_{}_{}'.format(num_episodes, int(self.env.unwrapped.human_num))
        if self.load_weights(weights_file):
            return
        if os.path.exists(replay_buffer_file):
            self.replay_buffer.load(replay_buffer_file)
        else:
            demonstrator = self.initialize_demonstrator()
            self.env.unwrapped.max_time = il_max_time
            episode = 0
            while True:
                observations = []
                effects = []
                done = False
                info = ''
                obs = self.env.reset()
                joint_state = self.env.unwrapped.compute_coordinate_observation(with_fov=True)
                while not done:
                    observations.append(obs)
                    demonstration = demonstrator.predict(joint_state)
                    target_action, action_class = self._approximate_action(demonstration)
                    obs, reward, done, info = self.env.step(target_action)
                    effects.append((torch.IntTensor([[action_class]]), reward, done))

                    if done:
                        logging.info(self.env.get_episode_summary() + ' in episode {}'.format(episode))
                        obs = self.env.reset()

                    joint_state = self.env.unwrapped.compute_coordinate_observation(with_fov=True)

                if info == 'Success':
                    episode += 1
                    for obs, effect in zip(observations, effects):
                        last_idx = self.replay_buffer.store_observation(obs)
                        self.replay_buffer.store_effect(last_idx, *effect)
                if episode > num_episodes:
                    break

            self.replay_buffer.save(replay_buffer_file)
            logging.info('Total steps: {}'.format(self.replay_buffer.num_in_buffer))

        # finish collecting experience and update the model
        if training == 'mc':
            criterion = nn.MSELoss().to(self.device)
            self._mc_update(optimizer, criterion, num_train_batch)
        elif training == 'classification':
            criterion = nn.CrossEntropyLoss().to(self.device)
            # self._action_classification_batch(optimizer, criterion, num_train_batch)
            self._action_classification_epoch(optimizer, criterion, num_epochs, step_size)
        elif training == 'regression':
            criterion = nn.MSELoss().to(self.device)
            self._action_classification_epoch(optimizer, criterion, num_epochs, step_size, regression=True)
        else:
            raise NotImplementedError

        torch.save(self.Q.state_dict(), weights_file)
        logging.info('Save imitation learning trained weights to {}'.format(weights_file))

        if test:
            self.test()

    def _approximate_action(self, demonstration):
        """ Approximate demonstration action with closest target action"""
        min_diff = float('inf')
        target_action = None
        index = -1
        for action_rot, value in self.action_dict.items():
            i, action_xy = value
            if isinstance(demonstration, ActionXY):
                diff = np.linalg.norm(np.array(action_xy) - np.array(demonstration))
            else:
                diff = np.linalg.norm(np.array(action_rot) - np.array(demonstration))
            if diff < min_diff:
                min_diff = diff
                target_action = action_rot
                index = i

        assert np.isclose(min_diff, 0)

        return target_action, index

    def initialize_demonstrator(self):
        model_dir = 'data/sarl'
        assert os.path.exists(model_dir)
        spec = importlib.util.spec_from_file_location('config', os.path.join(model_dir, 'config.py'))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        policy_config = config.PolicyConfig()
        policy = SARL()
        policy.epsilon = 0
        policy.configure(policy_config)
        policy.model.load_state_dict(torch.load(os.path.join(model_dir, 'rl_model.pth')))

        policy.set_device(torch.device('cpu'))
        policy.set_phase('test')
        policy.time_step = self.time_step

        return policy

    def test(self, visualize_step=False):
        logging.info('Start testing model')
        replay_buffer = ReplayBuffer(int(self.num_test_case * self.env.max_time / self.env.time_step),
                                     self.frame_history_len, self.image_size)

        num_saved_image = 0
        demonstrator = None
        cumulative_attention_diff = []
        cumulative_random_diff = []
        if visualize_step:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        for i in range(self.num_test_case):
            obs = self.env.reset()
            done = False

            episode_attention_diff = []
            episode_random_diff = []
            while not done:
                last_idx = replay_buffer.store_observation(obs)
                recent_observations = replay_buffer.encode_recent_observation()
                action = self.act(recent_observations)

                if demonstrator is None:
                    demonstrator = self.initialize_demonstrator()
                # compute the similarity of two attention models
                full_obs = self.env.unwrapped.compute_coordinate_observation()
                partial_obs, human_index_mapping = self.env.unwrapped.compute_coordinate_observation(True, True)
                _ = demonstrator.predict(partial_obs)
                demonstrator_attention = demonstrator.model.attention_weights

                # choose humans within FOV
                robot_state = full_obs.self_state
                human_states = full_obs.human_states
                human_directions = []
                in_view_humans = []
                for human_index, human_state in enumerate(human_states):
                    if human_state.px == 3 and human_state.py == -2:
                        # skip the dummy human
                        continue
                    angle = np.arctan2(human_state.py - robot_state.py, human_state.px - robot_state.px)
                    relative_angle = angle - robot_state.theta
                    if abs(relative_angle) < self.env.unwrapped.fov / 2 and human_index in human_index_mapping:
                        human_directions.append(
                            (relative_angle, demonstrator_attention[human_index_mapping[human_index]]))
                        in_view_humans.append((human_index, demonstrator_attention[human_index_mapping[human_index]]))

                # sort humans in the order of importance
                human_directions = sorted(human_directions, key=itemgetter(1), reverse=True)

                if self.Q.attention_weights is not None:
                    # compute the direction the agent is attending to
                    agent_attention = self.Q.attention_weights.squeeze().cpu().numpy()
                    max_cell_index = np.argmax(agent_attention)
                    horizontal_cell_index = max_cell_index % self.Q.W
                    # action_rot = self.env.unwrapped.actions[action.item()]
                    # logging.info('v: {:.2f}, r: {:.2f}'.format(action_rot[0], -np.rad2deg(action_rot[1])))
                else:
                    # compute diff of the direction of max response in cnn and the attention direction of demonstrator
                    horizontal_cell_index = self.Q.max_response_index.cpu().numpy()[0] % self.Q.W

                if visualize_step:
                    # plt.axis('scaled')
                    plt.ion()
                    # ax1.imshow(obs.image[:, :, 0], cmap='gray')
                    if self.Q.attention_weights is not None:
                        top_down_view(full_obs, ax1, in_view_humans)
                        attention_weights = self.Q.attention_weights.squeeze().view(7, 7).cpu().numpy()
                        ax2.axis('off')
                        heatmap(obs.image[:, :, 0], attention_weights, ax=ax2)
                        plt.savefig(os.path.join('data/saved/image_' + str(num_saved_image)))
                        num_saved_image += 1
                    plt.show()

                # compute the distance between two attention directions
                fov = self.env.unwrapped.fov
                cell_fov = fov / self.Q.W
                agent_attention_direction = -fov / 2 + (horizontal_cell_index - 0.5) * cell_fov
                random_direction = np.random.uniform(-fov / 2, fov / 2)
                if human_directions:
                    attention_diff = abs(human_directions[0][0] - agent_attention_direction)
                    random_diff = abs(human_directions[0][0] - random_direction)
                    episode_attention_diff.append(attention_diff)
                    episode_random_diff.append(random_diff)

                obs, reward, done, info = self.env.step(action.item())
                replay_buffer.store_effect(last_idx, action, reward, done)

            logging.info(self.env.get_episode_summary() + ' with attention diff: {:.2f} and random diff: {:.2f}, in episode {}'.
                         format(np.mean(episode_attention_diff), np.mean(episode_random_diff), i))
            cumulative_attention_diff.append(np.mean(episode_attention_diff))
            cumulative_random_diff.append(np.mean(episode_random_diff))

        logging.info(self.env.get_episodes_summary(num_last_episodes=self.num_test_case))
        logging.info('Average attention direction difference: {:.4f}'.format(np.mean(cumulative_attention_diff)))
        logging.info('Random attention direction difference: {:.4f}'.format(np.mean(cumulative_random_diff)))

    def test_all_models(self, visualize_step=False, gt_att_diff=0):
        demonstrator = self.initialize_demonstrator()
        replay_buffer = ReplayBuffer(int(self.num_test_case * self.env.max_time / self.env.time_step),
                                     self.frame_history_len, self.image_size)

        models_order = ['plain_cnn', 'gda_regressor', 'gdda_residual_regressor', 'plain_cnn_mean']
        models_to_test = {'plain_cnn': 'GD-Net', 'gdda_residual_regressor': 'GDDA-Net', 'gda_regressor': 'GDA-Net',
                          'plain_cnn_mean': 'plain_cnn_mean'}

        # load il models
        models = dict()
        dir_names = os.listdir(self.output_dir)
        for dir_name in dir_names:
            if dir_name not in model_factory:
                logging.warning('Can not recognize dir name: {}'.format(dir_name))
            elif dir_name not in models_to_test:
                continue
            else:
                model = model_factory[dir_name](self.input_arg, self.num_actions, regression=self.regression).to(self.device)
                model.load_state_dict(torch.load(os.path.join(self.output_dir, dir_name, 'il_model.pth')))
                models[dir_name] = model
                logging.info('{} weights loaded'.format(dir_name))

        _, axes = plt.subplots(2, 2, figsize=(12, 12))
        # cumulative_saliency_diff = defaultdict(list)
        cumulative_random_diff = []
        cumulative_attention_acc = defaultdict(list)
        for case_num in range(self.num_test_case):
            obs = self.env.reset()
            done = False

            episode_attention_diff = defaultdict(list)
            # episode_saliency_diff = defaultdict(list)
            episode_random_diff = []
            episode_attention_acc = defaultdict(list)
            while not done:
                # compute the similarity of two attention models
                full_obs = self.env.unwrapped.compute_coordinate_observation()
                partial_obs, human_index_mapping = self.env.unwrapped.compute_coordinate_observation(True, True)
                demonstrator_action = demonstrator.predict(partial_obs)
                target_action, action_index = self._approximate_action(demonstrator_action)
                demonstrator_attention = demonstrator.model.attention_weights

                # choose humans within FOV
                robot_state = full_obs.self_state
                human_states = full_obs.human_states
                human_directions = []
                in_view_humans = []
                for human_index, human_state in enumerate(human_states):
                    if human_state.px == 3 and human_state.py == -2:
                        # skip the dummy human
                        continue
                    angle = np.arctan2(human_state.py - robot_state.py, human_state.px - robot_state.px)
                    relative_angle = angle - robot_state.theta
                    if abs(relative_angle) < self.env.unwrapped.fov / 2 and human_index in human_index_mapping:
                        human_directions.append(
                            (relative_angle, demonstrator_attention[human_index_mapping[human_index]]))
                        in_view_humans.append(
                            (human_index, demonstrator_attention[human_index_mapping[human_index]]))

                # sort humans in the order of importance
                human_directions = sorted(human_directions, key=itemgetter(1), reverse=True)

                for name in models_order:
                    model = models[name]
                    # compute observation for models
                    last_idx = replay_buffer.store_observation(obs)
                    recent_observations = replay_buffer.encode_recent_observation()
                    action = self.act(recent_observations, model)

                    if model.attention_weights is not None:
                        # compute the direction the agent is attending to
                        agent_attention = model.attention_weights.squeeze().cpu().numpy()
                        max_cell_index = np.argmax(agent_attention)
                        horizontal_cell_index = max_cell_index % model.W
                    else:
                        # horizontal_cell_index = model.max_response_index.cpu().numpy()[0] % model.W

                        # Get gradients
                        if not model.guided_backprop_initialized:
                            model.init_guided_backprop()
                        frames = torch.from_numpy(recent_observations[0]).unsqueeze(0).to(self.device) / 255.0
                        goals = torch.from_numpy(np.array(recent_observations[1])).unsqueeze(0).to(self.device)
                        grads = model.generate_gradients(frames, goals, action)
                        # plt.imshow(grads, cmap='gray')
                        # plt.show()
                        weights = np.zeros((model.H, model.W))
                        scale = int(84 / model.H)
                        for i in range(model.H):
                            for j in range(model.W):
                                weights[i, j] = np.nansum(grads[i*scale:(i+1)*scale, j*scale:(j+1)*scale], )
                        weights = np.reshape(weights, (1, model.H * model.W))
                        weights = torch.nn.functional.softmax(torch.from_numpy(weights).squeeze(), dim=0).numpy()
                        max_cell_index = np.argmax(weights)
                        horizontal_cell_index = max_cell_index % model.W

                    # compute the distance between two attention directions
                    fov = self.env.unwrapped.fov
                    cell_fov = fov / model.W
                    agent_attention_direction = -fov / 2 + (horizontal_cell_index - 0.5) * cell_fov
                    # if model.attention_weights is None:
                    #     saliency_attention_direction = -fov / 2 + (saliency_horizontal_cell_index - 0.5) * cell_fov

                    if human_directions:
                        if (len(human_directions) > 1) and (
                                (human_directions[0][1] - human_directions[1][1]) < gt_att_diff):
                            pass
                        else:
                            attention_diff = abs(human_directions[0][0] - agent_attention_direction)
                            episode_attention_diff[name].append(attention_diff)
                            # if model.attention_weights is None:
                            #     saliency_diff = abs(human_directions[0][0] - saliency_attention_direction)
                            #     episode_saliency_diff[name].append(saliency_diff)

                            # compute attention accuracy
                            if attention_diff < cell_fov:
                                episode_attention_acc[name].append(1)
                            else:
                                episode_attention_acc[name].append(0)

                # compute random attention direction
                if human_directions:
                    if (len(human_directions) > 1) and ((human_directions[0][1] - human_directions[1][1]) < gt_att_diff):
                        pass
                    else:
                        random_direction = np.random.uniform(-fov / 2, fov / 2)
                        random_diff = abs(human_directions[0][0] - random_direction)
                        episode_random_diff.append(random_diff)

                if visualize_step:
                    # plt.axis('scaled')
                    plt.ion()
                    axes[0][0].axis('off')
                    # axes[0][0].imshow(obs.image[:, :, 0], cmap='gray')
                    top_down_view(full_obs, axes[0][0], in_view_humans)
                    index = 1
                    for plot_name in models_order:
                        model = models[plot_name]
                        if plot_name == 'plain_cnn_mean':
                            continue

                        ax = axes[int(index / 2)][index % 2]
                        ax.axis('off')
                        if model.attention_weights is not None:
                            attention_weights = model.attention_weights.squeeze().view(7, 7).cpu().numpy()
                            heatmap(obs.image[:, :, 0], attention_weights, ax=ax)
                        else:
                            # mean_block_response = model.mean_block_response.squeeze().view(7, 7).cpu().numpy()
                            # heatmap(obs.image[:, :, 0], mean_block_response, ax=ax)
                            if not model.guided_backprop_initialized:
                                model.init_guided_backprop()
                            frames = torch.from_numpy(recent_observations[0]).unsqueeze(0).to(self.device) / 255.0
                            goals = torch.from_numpy(np.array(recent_observations[1])).unsqueeze(0).to(self.device)
                            grads = model.generate_gradients(frames, goals, action)
                            # plt.imshow(grads, cmap='gray')
                            # plt.show()
                            weights = np.zeros((model.H, model.W))
                            scale = int(84 / model.H)
                            for i in range(model.H):
                                for j in range(model.W):
                                    weights[i, j] = np.nansum(
                                        grads[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale], )
                            weights = np.reshape(weights, (1, model.H * model.W))
                            weights = torch.nn.functional.softmax(torch.from_numpy(weights).squeeze(), dim=0).view(7, 7).numpy()
                            heatmap(obs.image[:, :, 0], weights, ax=ax)
                        ax.set_title(models_to_test[plot_name])

                        index += 1
                    plt.show()

                obs, reward, done, info = self.env.step(target_action)
                # replay_buffer.store_effect(last_idx, action, reward, done)

            if info == 'Overtime':
                # skip overtime experience
                continue

            logging.info(self.env.get_episode_summary() + ' in episode {}'.format(case_num))
            for model in models:
                cumulative_attention_diff[model].append(np.mean(episode_attention_diff[model]))
            # for name, saliency_diff in episode_saliency_diff.items():
            #     cumulative_saliency_diff[name].append(np.mean(saliency_diff))
            cumulative_random_diff.append(np.mean(episode_random_diff))
            for name, attention_acc in episode_attention_acc.items():
                cumulative_attention_acc[name] += attention_acc

        logging.info(self.env.get_episodes_summary(num_last_episodes=self.num_test_case))
        for name, attention_diff in cumulative_attention_diff.items():
            logging.info('{:<40} attention direction difference: {:.4f} averaged over {}'.
                         format(name, np.mean(attention_diff), len(attention_diff)))
        # for name, saliency_diff in cumulative_saliency_diff.items():
        #     logging.info('{:<40} saliency direction difference: {:.4f} averaged over {}'.
        #                  format(name, np.mean(saliency_diff), len(saliency_diff)))
        logging.info('Random attention direction difference: {:.4f}'.format(np.mean(cumulative_random_diff)))
        for name, attention_acc in cumulative_attention_acc.items():
            logging.info('{:<40} attention accuracy: {:.4f}'.format(name, np.mean(attention_acc)))

    def reinforcement_learning(self, optimizer_spec, exploration, learning_starts=50000,
                               learning_freq=4, num_timesteps=2000000, episode_update=False):
        statistics_file = os.path.join(self.output_dir, 'statistics.json')
        weights_file = os.path.join(self.output_dir, 'rl_model.pth')
        self.load_weights(weights_file)
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

    def _action_classification_batch(self, optimizer, criterion, num_train_batch):
        for _ in range(num_train_batch):
            frames_batch, goals_batch, action_batch, _, _, _, done_mask = \
                self.replay_buffer.sample(self.batch_size)
            # Convert numpy nd_array to torch variables for calculation
            frames_batch = torch.from_numpy(frames_batch).to(self.device) / 255.0
            goals_batch = torch.from_numpy(goals_batch).to(self.device)
            action_batch = torch.from_numpy(action_batch).long().to(self.device)

            predicted_actions = self.Q(frames_batch, goals_batch)
            loss = criterion(predicted_actions, action_batch)
            optimizer.zero_grad()
            loss.backward()

            # Perform the update
            optimizer.step()
            self.num_param_updates += 1

            if self.num_param_updates % self.log_every_n_steps == 0:
                logging.info('Batch loss: {:.4f} after {} batches'.format(loss.item(), self.num_param_updates))

    def _action_classification_epoch(self, optimizer, criterion, num_train_epochs, step_size, regression=False):
        # construct dataloader and store experiences in dataset
        datasets = {split: BufferWrapper(self.replay_buffer, split) for split in ['train', 'val', 'test']}
        dataloaders = {split: DataLoader(datasets[split], self.batch_size, shuffle=True, collate_fn=pack_batch)
                       for split in ['train', 'val', 'test']}
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        model = self.Q
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        best_acc = 0.0

        for epoch in range(num_train_epochs):
            logging.info('-' * 10)
            logging.info('Epoch {}/{}'.format(epoch, num_train_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train' and epoch != 0:
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterating over data once is one epoch
                for data in dataloaders[phase]:
                    # get the inputs
                    frames_batch, goals_batch, class_batch = data
                    frames_batch = frames_batch.to(self.device) / 255.0
                    goals_batch = goals_batch.to(self.device)
                    class_batch = class_batch.long().to(self.device)
                    if regression:
                        # action_batch = action_batch.long().to(self.device)
                        action_batch = self.idx2action[class_batch.cpu()].to(self.device)
                    else:
                        action_batch = class_batch

                    # zero the parameter gradients and forward
                    optimizer.zero_grad()
                    outputs = model(frames_batch, goals_batch)
                    loss = criterion(outputs, action_batch)
                    if regression:
                        class_losses = []
                        for idx in range(self.idx2action.size(0)):
                            actions = self.idx2action[idx].to(self.device).unsqueeze(0).expand(outputs.size(0), 2)
                            class_loss = torch.nn.functional.mse_loss(outputs, actions, reduce=False)
                            class_losses.append(torch.sum(class_loss, 1, keepdim=True))
                        class_losses = torch.cat(class_losses, 1)
                        _, preds = torch.min(class_losses.data, 1)
                    else:
                        _, preds = torch.max(outputs.data, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch != 0:
                        loss.backward()
                        optimizer.step()
                    # statistics
                    running_loss += loss.data.item() * frames_batch.size(0)
                    running_corrects += torch.sum(preds == class_batch.data).item()
                epoch_loss = running_loss / len(datasets[phase])
                epoch_acc = running_corrects / len(datasets[phase])
                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if self.use_best_wts:
                        best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val loss: {:4f}'.format(best_loss))
        # load best model weights
        if self.use_best_wts:
            model.load_state_dict(best_model_wts)

        # test model
        phase = 'test'
        model.train(False)
        running_loss = 0.0
        running_corrects = 0
        # Iterating over data once is one epoch
        for data in dataloaders[phase]:
            # get the inputs
            frames_batch, goals_batch, class_batch = data
            frames_batch = frames_batch.to(self.device) / 255.0
            goals_batch = goals_batch.to(self.device)
            class_batch = class_batch.long().to(self.device)
            if regression:
                # action_batch = action_batch.long().to(self.device)
                action_batch = self.idx2action[class_batch.cpu()].to(self.device)
            else:
                action_batch = class_batch

            # zero the parameter gradients and forward
            optimizer.zero_grad()
            outputs = model(frames_batch, goals_batch)
            loss = criterion(outputs, action_batch)
            if regression:
                class_losses = []
                for idx in range(self.idx2action.size(0)):
                    actions = self.idx2action[idx].to(self.device).unsqueeze(0).expand(outputs.size(0), 2)
                    class_loss = torch.nn.functional.mse_loss(outputs, actions, reduce=False)
                    class_losses.append(torch.sum(class_loss, 1, keepdim=True))
                class_losses = torch.cat(class_losses, 1)
                _, preds = torch.min(class_losses.data, 1)
            else:
                _, preds = torch.max(outputs.data, 1)

            # statistics
            running_loss += loss.data.item() * frames_batch.size(0)
            running_corrects += torch.sum(preds == class_batch.data).item()
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects / len(datasets[phase])

        logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        return model, best_loss

    def load_weights(self, weights_file):
        if os.path.exists(weights_file):
            self.Q.load_state_dict(torch.load(weights_file))
            self.target_Q.load_state_dict(torch.load(weights_file))
            logging.info('Imitation learning trained weight loaded')
            return True
        else:
            return False


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--model', type=str, default='dqn')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--with_il', default=True, action='store_true')
    parser.add_argument('--il_training', type=str, default='classification')
    parser.add_argument('--num_episodes', type=int, default=4000)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--use_best_wts', default=True, action='store_true')
    parser.add_argument('--step_size', type=int, default=150)
    parser.add_argument('--frame_history_len', type=int, default=1)
    parser.add_argument('--gt_att_diff', type=float, default=0)
    parser.add_argument('--with_rl', default=False, action='store_true')
    parser.add_argument('--eps_start', type=float, default=1)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay_steps', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_timesteps', type=int, default=2000000)
    parser.add_argument('--learning_starts', type=int, default=50000)
    parser.add_argument('--reward_shaping', default=False, action='store_true')
    parser.add_argument('--curriculum_learning', default=False, action='store_true')
    parser.add_argument('--episode_update', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_il', default=False, action='store_true')
    parser.add_argument('--test_rl', default=False, action='store_true')
    parser.add_argument('--test_all_models', default=False, action='store_true')
    parser.add_argument('--num_test_case', type=int, default=200)
    parser.add_argument('--visualize_step', default=False, action='store_true')
    args = parser.parse_args()

    if args.test_il or args.test_rl or args.test_all_models:
        if not os.path.exists(args.output_dir):
            raise ValueError('Model dir does not exist')
    else:
        # configure paths
        make_new_dir = True
        if os.path.exists(args.output_dir):
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y':
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
        if make_new_dir:
            os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    monitor_output_dir = os.path.join(args.output_dir, 'monitor-outputs')

    # configure logging
    file_handler = logging.FileHandler(log_file, mode='a')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(sys.argv)
    if not args.test_il and not args.test_rl:
        logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
        logging.info('Using device: %s', device)
        logging.info(pprint.pformat(vars(args), indent=4))

    # configure environment
    env = VisualSim(reward_shaping=args.reward_shaping, curriculum_learning=args.curriculum_learning)
    env = MyMonitor(env, monitor_output_dir)
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    trainer = Trainer(
        env=env,
        q_func=model_factory[args.model],
        device=device,
        output_dir=args.output_dir,
        replay_buffer_size=100000,
        batch_size=args.batch_size,
        gamma=args.gamma,
        frame_history_len=args.frame_history_len,
        target_update_freq=10000,
        num_test_case=args.num_test_case,
        use_best_wts=args.use_best_wts,
        regression=(args.il_training == 'regression')
    )

    if args.test_il:
        trainer.load_weights(os.path.join(args.output_dir, 'il_model.pth'))
        trainer.test(args.visualize_step)
    elif args.test_rl:
        trainer.load_weights(os.path.join(args.output_dir, 'rl_model.pth'))
        trainer.test(args.visualize_step)
    elif args.test_all_models:
        trainer.test_all_models(visualize_step=args.visualize_step,
                                gt_att_diff=args.gt_att_diff)
    else:
        # imitation learning
        if args.with_il:
            trainer.imitation_learning(
                num_episodes=args.num_episodes,
                training=args.il_training,
                num_epochs=args.num_epochs,
                step_size=args.step_size,
                test=args.test
            )

        # reinforcement learning
        if args.with_rl:
            rl_optimizer_spec = OptimizerSpec(
                constructor=optim.RMSprop,
                kwargs=dict(lr=0.00025, alpha=0.95, eps=0.01),
            )
            if args.eps_decay_steps == 0:
                exploration_schedule = ConstantSchedule(args.eps_end)
                logging.info('Use constant exploration rate: {}'.format(args.eps_end))
            else:
                exploration_schedule = LinearSchedule(args.eps_decay_steps, args.eps_end, args.eps_start)
            trainer.reinforcement_learning(
                optimizer_spec=rl_optimizer_spec,
                exploration=exploration_schedule,
                learning_starts=args.learning_starts,
                learning_freq=4,
                num_timesteps=args.num_timesteps,
                episode_update=args.episode_update
            )


if __name__ == '__main__':
    main()
