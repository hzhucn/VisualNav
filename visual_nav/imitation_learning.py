import time
import copy
import logging
import os
import importlib.util
from operator import itemgetter
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.sarl import SARL
from visual_nav.utils.buffers import DemoBuffer, BufferWrapper, pack_batch
from visual_nav.utils.visualization_tools import heatmap, top_down_view
from visual_nav.models.models import model_factory


class ImitationLearner(object):
    def __init__(self, env, model, device, output_dir, config):
        self.env = env
        self.device = device
        self.config = config
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.frame_history_len = config.frame_history_len
        self.target_update_freq = config.target_update_freq
        self.output_dir = output_dir
        self.num_test_case = config.num_test_case
        self.use_best_wts = config.use_best_wts
        self.regression = config.regression

        img_h, img_w, img_c = env.observation_space.shape
        self.input_arg = config.frame_history_len * img_c
        self.num_actions = env.action_space.n
        self.image_size = (img_h, img_w, img_c)
        self.time_step = env.unwrapped.time_step

        self.model = model(self.input_arg, self.num_actions, regression=config.regression).to(device)
        self.target_model = model(self.input_arg, self.num_actions, regression=config.regression).to(device)
        # self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, self.image_size)
        self.replay_buffer = None

        self.log_every_n_steps = 10000
        self.num_param_updates = 0
        # map action_rot to its index and action_xy
        self.action_dict = {action: (i, ActionXY(action.v * np.cos(action.r), action.v * np.sin(action.r)))
                            for i, action in enumerate(self.env.unwrapped.actions)}
        self.idx2action = torch.from_numpy(np.stack([action for action in self.env.unwrapped.actions])).float()

    def _collect_demonstration(self, num_episodes):
        demonstrator = self._initialize_demonstrator()
        episode = 0
        while True:
            observations = []
            effects = []
            robot_states = []
            human_states = []
            human_masks = []
            attention_weights = []
            done = False
            info = ''
            obs = self.env.reset()
            joint_state, human_mask = self.env.unwrapped.compute_coordinate_observation(with_fov=True)
            while not done:
                observations.append(obs)
                robot_states.append(joint_state.self_state)
                human_states.append(joint_state.human_states)
                human_masks.append(human_mask)

                # masking invisible humans
                joint_state.human_states = [joint_state.human_states[index] for index, mask in enumerate(human_mask) if mask]
                demonstration = demonstrator.predict(joint_state)
                attention_weights.append(demonstrator.get_attention_weights())
                target_action, action_class = self._approximate_action(demonstration)
                obs, reward, done, info = self.env.step(target_action)
                effects.append((torch.IntTensor([[action_class]]), reward, done))

                if done:
                    logging.info(self.env.get_episode_summary() + ' in episode {}'.format(episode))
                    obs = self.env.reset()

                joint_state, human_mask = self.env.unwrapped.compute_coordinate_observation(with_fov=True)

            if info == 'Success':
                episode += 1
                for i in range(len(observations)):
                    last_idx = self.replay_buffer.store_observation(observations[i])
                    self.replay_buffer.store_effect(last_idx, *effects[i])
                    self.replay_buffer.store_ground_truth_info(last_idx, robot_states[i], human_states[i],
                                                               human_masks[i], attention_weights[i])

                    # visualize frames
                    # import matplotlib.pyplot as plt
                    # fig, axes = plt.subplots(2, 2)
                    # axes[0, 0].imshow(obs.image[:, :, 0], cmap='gray')
                    # axes[0, 1].imshow(obs.image[:, :, 1], cmap='gray')
                    # axes[1, 0].imshow(obs.image[:, :, 2], cmap='gray')
                    # axes[1, 1].imshow(obs.image[:, :, 3], cmap='gray')
            if episode > num_episodes:
                break

    def train(self):
        """
        Imitation learning and reinforcement learning share the same environment, replay buffer and model function

        """
        num_episodes = self.config.num_episodes
        loss = self.config.loss
        num_epochs = self.config.num_epochs
        step_size = self.config.step_size
        max_time = self.env.unwrapped.max_time
        self.replay_buffer = DemoBuffer(int(num_episodes * max_time / self.env.time_step),
                                            self.frame_history_len, self.image_size)

        logging.info('Start imitation learning')
        weights_file = os.path.join(self.output_dir, 'il_model.pth')
        replay_buffer_file = 'data/replay_buffer_{}_{}'.format(num_episodes, int(self.env.unwrapped.human_num))
        if self.load_weights(weights_file):
            return
        if os.path.exists(replay_buffer_file):
            self.replay_buffer.load(replay_buffer_file)
            self.replay_buffer.preprocess()
        else:
            self._collect_demonstration(num_episodes)
            self.replay_buffer.save(replay_buffer_file)
            logging.info('Total steps: {}'.format(self.replay_buffer.num_in_buffer))
            self.replay_buffer.preprocess()

        # Train classification model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss().to(self.device)
            self._train_classification(optimizer, criterion, num_epochs, step_size)
        elif loss == 'regression':
            criterion = nn.MSELoss().to(self.device)
            self._train_classification(optimizer, criterion, num_epochs, step_size, regression=True)
        else:
            raise NotImplementedError

        torch.save(self.model.state_dict(), weights_file)
        logging.info('Save imitation learning trained weights to {}'.format(weights_file))

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

    def _initialize_demonstrator(self):
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

    def act(self, obs, model=None):
        frames = torch.from_numpy(obs[0]).unsqueeze(0).to(self.device) / 255.0
        goals = torch.from_numpy(np.array(obs[1])).unsqueeze(0).to(self.device)
        if model:
            return model(frames, goals).data.max(1)[1].cpu()
        else:
            return self.model(frames, goals).data.max(1)[1].cpu()

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
                    demonstrator = self._initialize_demonstrator()
                # compute the similarity of two attention models
                full_obs, _ = self.env.unwrapped.compute_coordinate_observation()
                partial_obs, human_mask = self.env.unwrapped.compute_coordinate_observation(True)
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

                if self.model.attention_weights is not None:
                    # compute the direction the agent is attending to
                    agent_attention = self.model.attention_weights.squeeze().cpu().numpy()
                    max_cell_index = np.argmax(agent_attention)
                    horizontal_cell_index = max_cell_index % self.model.W
                    # action_rot = self.env.unwrapped.actions[action.item()]
                    # logging.info('v: {:.2f}, r: {:.2f}'.format(action_rot[0], -np.rad2deg(action_rot[1])))
                else:
                    # compute diff of the direction of max response in cnn and the attention direction of demonstrator
                    horizontal_cell_index = self.model.max_response_index.cpu().numpy()[0] % self.model.W

                if visualize_step:
                    # plt.axis('scaled')
                    plt.ion()
                    # ax1.imshow(obs.image[:, :, 0], cmap='gray')
                    if self.model.attention_weights is not None:
                        top_down_view(full_obs, ax1, in_view_humans)
                        attention_weights = self.model.attention_weights.squeeze().view(7, 7).cpu().numpy()
                        ax2.axis('off')
                        heatmap(obs.image[:, :, 0], attention_weights, ax=ax2)
                        plt.savefig(os.path.join('data/saved/image_' + str(num_saved_image)))
                        num_saved_image += 1
                    plt.show()

                # compute the distance between two attention directions
                fov = self.env.unwrapped.fov
                cell_fov = fov / self.model.W
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
        demonstrator = self._initialize_demonstrator()
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
        cumulative_attention_diff = defaultdict(list)
        for case_num in range(self.num_test_case):
            obs = self.env.reset()
            done = False

            episode_attention_diff = defaultdict(list)
            # episode_saliency_diff = defaultdict(list)
            episode_random_diff = []
            episode_attention_acc = defaultdict(list)
            while not done:
                # compute the similarity of two attention models
                full_obs, _ = self.env.unwrapped.compute_coordinate_observation()
                partial_obs, human_mask = self.env.unwrapped.compute_coordinate_observation(True)
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

    def _train_classification(self, optimizer, criterion, num_train_epochs, step_size, regression=False):
        # construct dataloader and store experiences in dataset
        datasets = {split: BufferWrapper(self.replay_buffer, split) for split in ['train', 'val', 'test']}
        dataloaders = {split: DataLoader(datasets[split], self.batch_size, shuffle=True, collate_fn=pack_batch)
                       for split in ['train', 'val', 'test']}
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

        model = self.model
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')

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
            self.model.load_state_dict(torch.load(weights_file))
            self.target_model.load_state_dict(torch.load(weights_file))
            logging.info('Imitation learning trained weight loaded')
            return True
        else:
            return False
