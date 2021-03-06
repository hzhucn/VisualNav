import logging
import configparser
import os
import argparse
import pprint
import torch
import gym
from visual_sim.envs.visual_sim import VisualSim
from crowd_nav.policy.sarl import SARL


def test():
    parser = argparse.ArgumentParser('Parse test configuration')
    parser.add_argument('--num_test_case', type=int, default=50)
    parser.add_argument('--human_num', type=int, default=4)
    parser.add_argument('--with_fov', default=False, action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info(pprint.pformat(args))

    env = gym.make('VisualSim-v0')
    env = VisualSim()
    env.human_num = args.human_num

    # configure SARL
    model_dir = 'crowdnav_data/orca_square_20p_nearest_10p_invisible/sarl_without_global_state'
    assert os.path.exists(model_dir)
    policy = SARL()
    policy.epsilon = 0
    policy_config = configparser.RawConfigParser()
    policy_config.read(os.path.join(model_dir, 'policy.config'))
    policy.configure(policy_config)
    policy.model.load_state_dict(torch.load(os.path.join(model_dir, 'rl_model.pth')))

    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')
    policy.time_step = env.time_step

    success = 0
    collision = 0
    overtime = 0
    time = []
    num_test_case = args.num_test_case
    for i in range(num_test_case):
        ob = env.reset()
        joint_state = env.compute_coordinate_observation(args.with_fov)
        done = False
        while not done:
            action = policy.predict(joint_state)
            ob, reward, done, info = env.step(action)
            joint_state = env.compute_coordinate_observation(args.with_fov)
            if info == 'Success':
                success += 1
                time.append(env.time)
            elif info == 'Collision':
                collision += 1
            elif info == 'Overtime':
                overtime += 1

        logging.info('Episode ends with signal: {} in {}s'.format(info, env.time))

    avg_time = sum(time) / len(time) if time else 0
    logging.info('Success: {:.2f}, collision: {:.2f}, overtime: {:.2f}, average time: {:.2f}s'.format(
        success/num_test_case, collision/num_test_case, overtime/num_test_case, avg_time))


if __name__ == '__main__':
    test()
