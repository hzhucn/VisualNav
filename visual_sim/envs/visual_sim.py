import math

import numpy as np
from numpy.linalg import norm
from gym import Env
import airsim


class VisualSim(Env):
    def __init__(self):
        client = airsim.CarClient()
        client.confirmConnection()
        client.enableApiControl(True)
        self.client = client
        self.initial_position = None
        self.goal_position = None
        self.distance = 10000

    def reset(self):
        self.client.reset()
        position = self.client.getCarState().kinematics_estimated.position
        self.initial_position = np.array((position.x_val, position.y_val, position.z_val))
        self.goal_position = self.initial_position + (self.distance, 0, 0)

    def step(self, action):
        car_controls = interpret_action(action)
        self.client.setCarControls(car_controls)

        car_state = self.client.getCarState()
        position = car_state.kinematics_estimated.position
        current_position = np.array((position.x_val, position.y_val, position.z_val))
        dist = norm(current_position - self.goal_position)
        collision_info = self.client.simGetCollisionInfo()
        print(collision_info)
        if dist < 10:
            reward = 1
            done = True
            info = 'Accomplishment'
        elif collision_info.has_collided:
            reward = -1
            done = True
            info = 'Collision'
        else:
            reward = 0
            done = False
            info = ''

        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
                                              airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True, False),
                                              airsim.ImageRequest("2", airsim.ImageType.DepthPerspective, True, False)])
        observation = transform_input(responses)

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def compute_reward(self, car_state):
        # MAX_SPEED = 300
        # MIN_SPEED = 10
        # thresh_dist = 3.5
        # beta = 3
        #
        # z = 0
        # pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]),
        #        np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]),
        #        np.array([0, -1, z])]
        # pd = car_state.kinematics_estimated.position
        # car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])
        #
        # dist = 10000000
        # for i in range(0, len(pts) - 1):
        #     dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))) / np.linalg.norm(
        #         pts[i] - pts[i + 1]))
        #
        # # print(dist)
        # if dist > thresh_dist:
        #     reward = -3
        # else:
        #     reward_dist = (math.exp(-beta * dist) - 0.5)
        #     reward_speed = (((car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) - 0.5)
        #     reward = reward_dist + reward_speed



        return reward


def interpret_action(action):
    car_controls = airsim.CarControls()
    car_controls.brake = 0
    car_controls.throttle = 1
    if action == 0:
        car_controls.throttle = 0
        car_controls.brake = 1
    elif action == 1:
        car_controls.steering = 0
    elif action == 2:
        car_controls.steering = 0.5
    elif action == 3:
        car_controls.steering = -0.5
    elif action == 4:
        car_controls.steering = 0.25
    else:
        car_controls.steering = -0.25
    return car_controls


def isDone(car_state, car_controls, reward):
    # done = 0
    # if reward == -1:
    #     done = 1
    # if car_controls.brake == 0:
    #     if car_state.speed <= 5:
    #         done = 1

    if reward == -1 or reward == 1:
        done = True
    else:
        done = False

    return done


def transform_input(responses):
    images = []
    for response in responses:
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image
        image = Image.fromarray(img2d)
        # image = np.array(image.resize((84, 84)).convert('L'))
        images.append(image)

    return images
