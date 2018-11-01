import logging
import argparse
import torch
import gym
from visual_sim.envs.visual_sim import VisualSim
from visual_nav.utils.robot import Robot
from visual_nav.utils.explorer import Explorer


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
    explorer = Explorer(env, robot, device)

    if args.visualize:
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