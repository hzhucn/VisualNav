import logging
import configparser
import torch
import gym
from visual_sim.envs.visual_sim import VisualSim
from crowd_nav.policy.sarl import SARL


def test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    env = gym.make('VisualSim-v0')
    env = VisualSim()

    # configure SARL
    policy = SARL()
    policy.epsilon = 0
    policy_config = configparser.RawConfigParser()
    policy_config.read('data/sarl/policy.config')
    policy.configure(policy_config)
    policy.model.load_state_dict(torch.load('data/sarl/rl_model.pth'))

    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')
    policy.time_step = env.time_step

    success = 0
    collision = 0
    overtime = 0
    time = []
    for i in range(env.test_case_num):
        ob = env.reset()
        joint_state = env.compute_coordinate_observation()
        done = False
        while not done:
            action = policy.predict(joint_state)
            ob, reward, done, info = env.step(action)
            joint_state = env.compute_coordinate_observation()
            if info == 'Accomplishment':
                success += 1
                time.append(env.time)
            elif info == 'Collision':
                collision += 1
            elif info == 'Overtime':
                overtime += 1

        logging.info('Episode ends with signal: {} in {}s'.format(info, env.time))

    avg_time = sum(time) / len(time) if time else 0
    logging.info('Success: {:.2f}, collision: {:.2f}, overtime: {:.2f}, average time: {:.2f}s'.format(
        success/env.test_case_num, collision/env.test_case_num, overtime/env.test_case_num, avg_time))


if __name__ == '__main__':
    test()