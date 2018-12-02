import numpy as np


class Config(object):
    pass


class ILConfig(object):
    replay_buffer_size = 1000000
    batch_size = 128
    gamma = 0.9
    frame_history_len = 1
    target_update_freq = 10000
    num_test_case = 100
    use_best_wts = False
    regression = False

    # training config
    num_episodes = 3000
    loss = 'cross_entropy'
    num_epochs = 50
    step_size = 100

    def __init__(self, debug=False):
        if debug:
            self.num_test_case = 2
            self.num_episodes = 10


class EnvConfig(object):
    human_num = 4
    image_type = 'DepthPerspective'
    reward_shaping = False
    curriculum_learning = False

    # action space
    speed_samples = 2
    rotation_samples = 7
    abs_rotation_bound = np.pi / 3

    num_frame_per_step = 4

    def __init__(self, debug=False):
        if debug:
            pass
