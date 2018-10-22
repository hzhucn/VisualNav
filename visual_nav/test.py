import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from visual_nav.utils.robot import Robot
from visual_sim.envs.visual_sim import VisualSim
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--video_file', type=str, default=None)

    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure variables
    env = gym.make('VisualSim-v0')
    env = VisualSim()
    robot = Robot()

    while True:
        ob = env.reset()
        done = False
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            plt.imshow(ob[1])
        logging.info('Episode ends with signal: {}'.format(info))

    if args.visualize:
        pass
    else:
        # explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        pass


if __name__ == '__main__':
    main()