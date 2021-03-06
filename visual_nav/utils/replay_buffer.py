"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import random
import os
import shutil
import logging

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(Dataset):
    def __init__(self, size, frame_history_len, image_size):
        """This is a memory efficient implementation of the replay buffer.

        The specific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.image_size = image_size

        self.next_idx = 0
        self.num_in_buffer = 0

        self.frames = None
        self.goals = None
        self.action = None
        self.reward = None
        self.done = None
        self.value = None

    def __len__(self):
        return self.num_in_buffer

    def __getitem__(self, idx):
        return self.encode_observation(idx)

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, with_value=False):
        frames_batch = []
        goals_batch = []
        for idx in idxes:
            frames, goals = self.encode_observation(idx)
            frames_batch.append(frames[np.newaxis, :])
            goals_batch.append(goals[np.newaxis, :])
        frames_batch = np.concatenate(frames_batch, 0)
        goals_batch = np.concatenate(goals_batch, 0)

        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]

        next_frames_batch = []
        next_goals_batch = []
        for idx in idxes:
            next_frames, next_goals = self.encode_observation(idx + 1)
            next_frames_batch.append(next_frames[np.newaxis, :])
            next_goals_batch.append(next_goals[np.newaxis, :])
        next_frames_batch = np.concatenate(next_frames_batch, 0)
        next_goals_batch = np.concatenate(next_goals_batch, 0)

        done_mask = self.done[idxes]

        if not with_value:
            return frames_batch, goals_batch, act_batch, rew_batch, next_frames_batch, next_goals_batch, done_mask
        else:
            value_batch = self.value[idxes]
            return frames_batch, goals_batch, act_batch, rew_batch, next_frames_batch, next_goals_batch, done_mask, \
                   value_batch

    def sample(self, batch_size, with_value=False):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes, with_value)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self.encode_observation((self.next_idx - 1) % self.size)

    def encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundary of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.frames[0]) for _ in range(missing_context)]
            goals = [np.zeros_like(self.goals[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.frames[idx % self.size])
                goals.append(self.goals[idx % self.size])
            frames = np.concatenate(frames, 0)
            goals = np.stack(goals)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.frames.shape[2], self.frames.shape[3]
            frames = self.frames[start_idx:end_idx].reshape(-1, img_h, img_w)
            goals = self.goals[start_idx:end_idx].reshape(-1, 2)

        return frames, goals

    def store_observation(self, ob):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        ob: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            and the frame will transpose to shape (img_h, img_w, img_c) to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        frame = ob.image
        if frame.shape != self.image_size:
            assert frame.shape[2] == 1
            frame = np.expand_dims(Image.fromarray(frame.squeeze()).resize(self.image_size[:2]), axis=2)
        goal = ob.goal

        # make sure we are not using low-dimensional observations, such as RAM
        if len(frame.shape) > 1:
            # transpose image frame into (img_c, img_h, img_w)
            frame = frame.transpose(2, 0, 1)

        if self.frames is None:
            self.frames = np.empty([self.size] + list(frame.shape), dtype=np.float32)
            self.goals = np.empty([self.size, 2], dtype=np.float32)
            self.action = np.empty([self.size], dtype=np.float32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.float32)
            self.value = np.empty([self.size], dtype=np.float32)

        self.frames[self.next_idx] = frame
        self.goals[self.next_idx] = np.array(goal)

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after observing frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that one can call `encode_recent_observation`
        in between.

        Parameters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def store_value(self, idx, value):
        self.value[idx] = value

    def save(self, output_dir):
        """ Save experience """
        if os.path.exists(output_dir):
            key = input('Replay buffer dir exists. Overwrite(y/n)?')
            if key == 'y':
                shutil.rmtree(output_dir)
            else:
                return
            os.mkdir(output_dir)
        else:
            os.mkdir(output_dir)

        np.save(os.path.join(output_dir, 'frames'), self.frames)
        np.save(os.path.join(output_dir, 'goals'), self.goals)
        np.save(os.path.join(output_dir, 'action'), self.action)
        np.save(os.path.join(output_dir, 'reward'), self.reward)
        np.save(os.path.join(output_dir, 'done'), self.done)
        np.save(os.path.join(output_dir, 'value'), self.value)
        with open(os.path.join(output_dir, 'num_in_buffer.txt'), 'w') as fo:
            fo.write(str(self.num_in_buffer))
        logging.info('Saved the replay buffer in {}'.format(output_dir))

    def load(self, input_dir):
        if not os.path.exists(input_dir):
            raise ValueError('Dir does not exist')

        self.frames = np.load(os.path.join(input_dir, 'frames.npy'))
        self.goals = np.load(os.path.join(input_dir, 'goals.npy'))
        self.action = np.load(os.path.join(input_dir, 'action.npy'))
        self.reward = np.load(os.path.join(input_dir, 'reward.npy'))
        self.done = np.load(os.path.join(input_dir, 'done.npy'))
        self.value = np.load(os.path.join(input_dir, 'value.npy'))

        with open(os.path.join(input_dir, 'num_in_buffer.txt'), 'r') as fo:
            self.num_in_buffer = int(fo.read())
            logging.info('The replay buffer loaded in {}'.format(input_dir))


class BufferWrapper(Dataset):
    def __init__(self, replay_buffer, split):
        self.replay_buffer = replay_buffer
        self.split = split

        # percentage range for different splits:
        split_percentages = {'train': (0, 0.7), 'val': (0.7, 0.8), 'test': (0.8, 1)}
        self.start_index = int(len(self.replay_buffer) * split_percentages[split][0])
        self.end_index = int(len(self.replay_buffer) * split_percentages[split][1])

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        buffer_idx = self.start_index + idx
        if self.replay_buffer.frame_history_len == 1:
            frames = self.replay_buffer.frames[buffer_idx]
            goals = self.replay_buffer.goals[buffer_idx]
            action = self.replay_buffer.action[buffer_idx]
        else:
            frames, goals = self.replay_buffer.encode_observation(buffer_idx)
            action = self.replay_buffer.action[buffer_idx]
        return frames, goals, action


def pack_batch(batch):
    frame_batch = torch.cat([torch.from_numpy(t[0]).unsqueeze(0) for t in batch], dim=0)
    goal_batch = torch.cat([torch.from_numpy(t[1]).unsqueeze(0) for t in batch], dim=0)
    action_batch = torch.cat([torch.Tensor([t[2]]) for t in batch], dim=0)

    return frame_batch, goal_batch, action_batch

