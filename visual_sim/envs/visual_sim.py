import itertools
from collections import defaultdict
from collections import namedtuple
import logging
from PIL import Image

import numpy as np
from numpy.linalg import norm
from gym import Env
from gym.spaces import Discrete, Box
import airsim
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from crowd_sim.envs.utils.action import ActionXY, ActionRot


Goal = namedtuple('Goal', ['r', 'phi'])
Observation = namedtuple('Observation', ['image', 'goal'])

ImageType = namedtuple('ImageType', ['index', 'as_float', 'channel_size'])

ImageInfo = {
    'Scene': ImageType(0, False, 3),
    'DepthPlanner': ImageType(1, True, 1),
    'DepthPerspective': ImageType(2, True, 1),
    'DepthVis': ImageType(3, False, 1),
    'DisparityNormalized': ImageType(4, False, 1),
    'Segmentation': ImageType(5, False, 3),
    'SurfaceNormals': ImageType(6, False, 3),
    'Infrared': ImageType(7, False, 3)
}


class VisualSim(Env):
    """
    Image types:
      Scene = 0,
      DepthPlanner = 1,
      DepthPerspective = 2,
      DepthVis = 3,
      DisparityNormalized = 4,
      Segmentation = 5,
      SurfaceNormals = 6,
      Infrared = 7
    """
    def __init__(self, image_type='DepthPerspective', step_penalty=0):
        self.robot_dynamics = False
        self.blocking = True
        self.time_step = 0.25
        self.clock_speed = 10
        self.time = 0
        self.initial_position = np.array((0, 0, -1))
        self.goal_position = np.array((10, 0, 0))

        # rewards
        self.step_penalty = step_penalty
        self.collision_penalty = -1
        self.success_reward = 1
        self.max_time = 30

        # human
        self.human_num = 2
        self.humans = defaultdict(list)
        # 2.7 * 0.73
        self.human_radius = 1

        # robot
        self.max_speed = 1
        self.robot_radius = 0.3
        self.robot = list()

        # action space
        self.speed_samples = 3
        self.rotation_samples = 7
        self.actions = self.build_action_space()
        self.action_space = Discrete(self.speed_samples * self.rotation_samples + 1)

        # observation_space
        self.image_type = image_type
        self.observation_space = Box(low=0, high=255, shape=(84, 84, ImageInfo[image_type].channel_size))

        # train test setting
        self.test_case_num = 5

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
        self.humans = defaultdict(list)
        self.robot = list()

        if self.robot_dynamics:
            self.client.reset()
        else:
            self.client.reset()

        return self.compute_observation()

    def step(self, action):
        import time
        pose = self.client.simGetVehiclePose()
        position = pose.position
        orientation = pose.orientation
        if self.robot_dynamics:
            car_controls = self.interpret_action(action)
            self.client.setCarControls(car_controls)
            if self.blocking:
                # pause for wall time self.time_step / self.clock_speed, which translates to game time self.time_step
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                while not self.client.simIsPause():
                    time.sleep(0.001)
        else:
            move = self.interpret_action(action)
            if isinstance(move, ActionXY):
                x = position.x_val + move.vx * self.time_step
                y = position.y_val + move.vy * self.time_step
                yaw = np.arctan2(y, x)
            elif isinstance(move, ActionRot):
                _, _, yaw = airsim.to_eularian_angles(orientation)
                yaw += move.r
                x = position.x_val + move.v * np.cos(yaw)
                y = position.y_val + move.v * np.sin(yaw)
            else:
                raise NotImplementedError
            self.move((x, y, self.initial_position[2]), yaw)
            if self.blocking:
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                # start = time.time()
                while not self.client.simIsPause():
                    time.sleep(0.001)
                # logging.debug('{:.2f}s past'.format(time.time() - start))
        self.time += self.time_step

        pose = self.client.simGetVehiclePose()
        if not (np.isclose(pose.position.x_val, x) and np.isclose(pose.position.y_val, y)):
            logging.debug('Different pose values between simGetVehiclePose and simSetVehiclePose!!!')
        reached_goal = self.distance_to_goal(pose.position) < self.robot_radius
        collision_info = self.client.simGetCollisionInfo()

        if reached_goal:
            reward = self.success_reward
            done = True
            info = 'Accomplishment'
        elif collision_info.has_collided:
            reward = self.collision_penalty
            done = True
            info = 'Collision'
        elif self.time > self.max_time:
            reward = 0
            done = True
            info = 'Overtime'
        else:
            reward = self.step_penalty
            done = False
            info = ''

        observation = self.compute_observation()

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def move(self, pos, yaw):
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(pos[0]), float(pos[1]), float(pos[2])),
                                                  airsim.to_quaternion(0, 0, yaw)), True)

    def compute_observation(self):
        # retrieve visual observation
        image_type = ImageInfo[self.image_type]
        responses = self.client.simGetImages([airsim.ImageRequest(0, image_type.index, image_type.as_float, False)])
        response = responses[0]

        if image_type.as_float:
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
            img2d = np.reshape(img1d, (response.height, response.width))
            image = np.expand_dims(Image.fromarray(img2d).resize(self.observation_space.shape[:2]).convert('L'), axis=2)
        else:
            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, image_type.channel_size)
            image = np.ascontiguousarray(image, dtype=np.uint8)

        # retrieve poses for both human and robot
        pose = self.client.simGetVehiclePose()
        r = self.distance_to_goal(pose.position)
        phi = np.arctan2(self.goal_position[1] - pose.position.y_val, self.goal_position[0] - pose.position.x_val)
        goal = Goal(r, phi)

        observation = Observation(image, goal)

        return observation

    def compute_coordinate_observation(self, fov=True):
        # retrieve all humans status
        for i in range(self.human_num):
            pose = self.client.simGetObjectPose('Human' + str(i))
            trials = 0
            while np.isnan(pose.position.x_val):
                pose = self.client.simGetObjectPose('Human' + str(i))
                trials += 1
                logging.debug('Get NaN pose value. Try to call API again...')

                if trials >= 3:
                    logging.warning('Cannot get human status from client. Check human_num.')
                    break
            self.humans[i].append(pose)
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
                return self.actions[action]
            elif isinstance(action, ActionRot) or isinstance(action, ActionXY):
                return action
            else:
                logging.error(action)

    def distance_to_goal(self, position):
        return norm((self.goal_position[0] - position.x_val, self.goal_position[1] - position.y_val))

    def build_action_space(self):
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_speed for i in
                  range(self.speed_samples)]
        rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        actions = [ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            actions.append(ActionRot(speed, rotation))

        return actions
