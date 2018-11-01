import math
import itertools
from collections import defaultdict
from collections import namedtuple

import numpy as np
from numpy.linalg import norm
from gym import Env
from gym.spaces import Discrete, Box
import airsim
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from crowd_sim.envs.utils.action import ActionXY, ActionRot


Goal = namedtuple('Goal', ['r', 'phi'])
Observation = namedtuple('Observation', ['image', 'goal'])


class VisualSim(Env):
    def __init__(self):
        self.initial_position = None
        self.goal_position = None
        self.robot_dynamics = False
        self.blocking = True
        self.time_step = 0.25
        self.clock_speed = 10
        self.time = 0

        # rewards
        self.collision_penalty = 0.2
        self.success_reward = 1
        self.max_time = 40
        self.goal_distance = 10

        # human
        self.human_num = 5
        self.humans = defaultdict(list)
        # 2.7 * 0.73
        self.human_radius = 1

        # robot
        self.max_speed = 1
        self.robot_radius = 0.3
        self.robot = list()

        # action space
        self.speed_samples = 3
        self.rotation_samples = 5
        self.actions = None
        self.action_space = Discrete(self.speed_samples * self.rotation_samples + 1)

        # observation_space
        self.image_types = [0]
        self.observation_space = Box(low=0, high=255, shape=(144, 256, 3))

        # train test setting
        self.test_case_num = 10

        if self.robot_dynamics:
            client = airsim.CarClient()
            client.enableApiControl(True)
            client.confirmConnection()
        else:
            client = airsim.VehicleClient()
        self.client = client
        if self.blocking:
            self.client.simPause(True)

    def reset(self):
        self.time = 0
        self.initial_position = np.array((0, 0, -1))
        self.goal_position = np.array((self.goal_distance, 0, 0))
        self.humans = defaultdict(list)
        self.robot = list()

        if self.robot_dynamics:
            self.client.reset()
        else:
            self.client.reset()
            # self.move(self.initial_position, 0)

        return self.compute_observation()

    def step(self, action):
        import time
        pose = self.client.simGetVehiclePose()
        if self.robot_dynamics:
            car_controls = self.interpret_action(action)
            self.client.setCarControls(car_controls)
            if self.blocking:
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                while not self.client.simIsPause():
                    time.sleep(0.01)
        else:
            move = self.interpret_action(action)
            if isinstance(move, ActionXY):
                x = pose.position.x_val + move.vx * self.time_step
                y = pose.position.y_val + move.vy * self.time_step
                yaw = np.arctan2(y, x)
            elif isinstance(move, ActionRot):
                yaw = move.r
                x = pose.position.x_val + move.v * np.cos(move.r)
                y = pose.position.y_val + move.v * np.sin(move.r)
            else:
                raise NotImplementedError
            self.move((x, y, self.initial_position[2]), yaw)
            if self.blocking:
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                while not self.client.simIsPause():
                    time.sleep(0.01)
        self.time += self.time_step

        position = pose.position
        current_position = np.array((position.x_val, position.y_val, 0))
        dist = norm(current_position - self.goal_position)
        collision_info = self.client.simGetCollisionInfo()
        if dist < self.robot_radius:
            reward = self.success_reward
            done = True
            info = 'Accomplishment'
        elif collision_info.has_collided:
            reward = self.collision_penalty
            done = True
            info = 'Collision'
        elif self.time >= self.max_time:
            reward = 0
            done = True
            info = 'Overtime'
        else:
            reward = 0
            done = False
            info = ''

        observation = self.compute_observation()

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def move(self, pos, yaw):
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(pos[0]), float(pos[1]), float(pos[2])),
                                                  airsim.to_quaternion(0, 0, yaw)), True)
        # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

    def compute_observation(self):
        # retrieve visual observation
        responses = self.client.simGetImages([airsim.ImageRequest("0", t, False, False) for t in self.image_types])
        images = []
        for image_type, response in zip(self.image_types, responses):
            if image_type in [2, 3]:
                # img1d = np.array(response.image_data_float, dtype=np.float)
                # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
                # image = np.reshape(img1d, (response.height, response.width)).astype(np.uint8)
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                image = img1d.reshape(response.height, response.width)
            elif image_type == 0:
                # get numpy array
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                # reshape array to 4 channel image array H X W X 4
                image = img1d.reshape(response.height, response.width, 4)[:, :, :3]
                image = np.ascontiguousarray(image, dtype=np.uint8)
            else:
                raise NotImplementedError

            images.append(image)

        # retrieve poses for both human and robot
        pose = self.client.simGetVehiclePose()
        r = norm((self.goal_position[0] - pose.position.x_val, self.goal_position[1] - pose.position.y_val))
        phi = np.arctan2(self.goal_position[1] - pose.position.y_val, self.goal_position[0] - pose.position.x_val)
        goal = Goal(r, phi)

        observation = Observation(images[0], goal)

        return observation

    def compute_coordinate_observation(self, in_fov=True):
        # retrieve all humans status
        for i in range(self.human_num):
            self.humans[i].append(self.client.simGetObjectPose('Human' + str(i)))
        self.robot.append(self.client.simGetVehiclePose())

        # Todo: only consider humans in FOV
        human_states = []
        for i in range(self.human_num):
            if len(self.humans[i]) == 1:
                vx = vy = 0
            else:
                vx = (self.humans[i][-1].position.x_val - self.humans[i][-2].position.x_val) / self.time_step
                vy = (self.humans[i][-1].position.y_val - self.humans[i][-2].position.y_val) / self.time_step
            px = self.humans[i][-1].position.x_val
            py = self.humans[i][-1].position.y_val
            human_state = ObservableState(px, py, vx, vy, self.human_radius)
            human_states.append(human_state)

        px = self.robot[-1].position.x_val
        py = self.robot[-1].position.y_val
        if len(self.robot) == 1:
            vx = vy = 0
        else:
            # TODO: use kinematics.linear_velocity?
            vx = self.robot[-1].position.x_val - self.robot[-2].position.x_val
            vy = self.robot[-1].position.y_val - self.robot[-2].position.y_val
        r  = self.robot_radius
        gx = self.goal_position[0]
        gy = self.goal_position[1]
        v_pref = 1
        _, _, theta = airsim.to_eularian_angles(self.robot[-1].orientation)
        robot_state = FullState(px, py, vx, vy, r, gx, gy, v_pref, theta)

        joint_state = JointState(robot_state, human_states)

        return joint_state

    def interpret_action(self, action):
        if self.robot_dynamics:
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
        else:
            if isinstance(action, int):
                if self.actions is None:
                    speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_speed for i in
                              range(self.speed_samples)]
                    rotations = np.linspace(-np.pi / 3, np.pi / 3, self.rotation_samples)

                    self.actions = [ActionRot(0, 0)]
                    for rotation, speed in itertools.product(rotations, speeds):
                        self.actions.append(ActionRot(speed, rotation))
                return self.actions[action]
            elif isinstance(action, ActionRot) or isinstance(action, ActionXY):
                return action
            else:
                print(action)
                raise NotImplementedError


