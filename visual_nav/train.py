import logging
import argparse
import configparser
import time
import cv2
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from visual_sim.envs.visual_sim import VisualSim
from visual_nav.utils.robot import Robot
from visual_nav.utils.detector import HOGDetector
# from visual_nav.utils.utils import coordinate_transform
from visual_nav.utils.explorer import Explorer


def main():
    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure variables
    env = gym.make('VisualSim-v0')
    env = VisualSim()

    while True:
        ob = env.reset()
        done = False
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
        logging.info('Episode ends with signal: {} in {}s'.format(info, env.time))
    else:
        explorer.run_k_episodes(env.test_case_num, 'test')


if __name__ == '__main__':
    main()