import sys
import logging
import os
import argparse
import shutil
import pprint
import importlib.util

import git
import gym
import gym.spaces
import torch

from visual_sim.envs.visual_sim import VisualSim
from visual_nav.utils.my_monitor import MyMonitor
from visual_nav.imitation_learning import ImitationLearner
from visual_nav.models.models import model_factory


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--model', type=str, default='plain_cnn')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--visualize_step', default=False, action='store_true')
    parser.add_argument('--training', type=str, default='il')

    args = parser.parse_args()

    # configure output folder
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.config = os.path.join(args.output_dir, args.config)
    spec = importlib.util.spec_from_file_location('config', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, args.output_dir)
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
    if not args.test:
        logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
        logging.info('Using device: %s', device)
        logging.info(pprint.pformat(vars(args), indent=4))

    # configure environment
    env = VisualSim(config=config.EnvConfig(args.debug))
    env = MyMonitor(env, monitor_output_dir)
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    if args.training == 'il':
        learner = ImitationLearner(env, model_factory[args.model], device, args.output_dir, config.ILConfig(args.debug))
    else:
        raise NotImplementedError

    if args.train:
        learner.train()
    if args.test:
        learner.load_weights(os.path.join(args.output_dir, 'il_model.pth'))
        learner.test(args.visualize_step)


if __name__ == '__main__':
    main()
