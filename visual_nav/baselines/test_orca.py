import logging
import torch
import gym
from visual_sim.envs.visual_sim import VisualSim
from crowd_sim.envs.policy.orca import ORCA


def test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    env = gym.make('VisualSim-v0')
    env = VisualSim()
    policy = ORCA()
    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')
    policy.time_step = env.time_step

    success = 0
    collision = 0
    overtime = 0
    time = []
    test_case_num = 10
    for i in range(test_case_num):
        ob = env.reset()
        joint_state = env.compute_coordinate_observation()
        done = False
        while not done:
            action = policy.predict(joint_state)
            ob, reward, done, info = env.step(action)
            # import matplotlib.pyplot as plt
            # plt.imshow(ob[0], cmap='gray')
            # plt.show()
            joint_state = env.compute_coordinate_observation()
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
        success/test_case_num, collision/test_case_num, overtime/test_case_num, avg_time))


if __name__ == '__main__':
    test()
