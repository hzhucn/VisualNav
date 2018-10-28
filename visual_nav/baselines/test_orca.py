from crowd_sim.envs.policy.orca import ORCA
import logging
import torch
import gym
from visual_sim.envs.visual_sim import VisualSim


def test():
    env = gym.make('VisualSim-v0')
    env = VisualSim()

    policy = ORCA()
    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')
    policy.time_step = env.time_step

    while True:
        ob = env.reset()
        joint_state = env.compute_coordinate_observation()
        done = False
        while not done:
            action = policy.predict(joint_state)
            ob, _, done, info = env.step(action)
            joint_state = env.compute_coordinate_observation()
        print('Episode ends with signal: {} in {}s'.format(info, env.time))
        logging.info('Episode ends with signal: {} in {}s'.format(info, env.time))


if __name__ == '__main__':
    test()