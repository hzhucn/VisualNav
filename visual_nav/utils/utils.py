import numpy as np
import random

import gym


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_wrapper_by_name(env, class_name):
    current_env = env
    while True:
        if class_name in current_env.__class__.__name__:
            return current_env
        elif isinstance(env, gym.Wrapper):
            current_env = current_env.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % class_name)
