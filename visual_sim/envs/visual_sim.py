import itertools
import logging
from collections import defaultdict
from collections import namedtuple

import airsim
import numpy as np
from PIL import Image
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from gym import Env
from gym.spaces import Discrete, Box
from numpy.linalg import norm

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
    def __init__(self, image_type='DepthPerspective', reward_shaping=False, curriculum_learning=False):
        self.robot_dynamics = False
        self.blocking = True
        self.time_step = 0.25
        self.clock_speed = 10
        self.time = 0
        self.initial_position = np.array((0, 0, -1))
        self.goal_distance = 6
        self.goal_position = np.array((self.goal_distance, 0, -1))

        # rewards
        self.collision_penalty = -0.25
        self.success_reward = 1
        self.max_time = 40
        self.reward_shaping = reward_shaping
        self.curriculum_learning = curriculum_learning
        self.early_reward_ratio = 0.5

        # human
        self.human_num = 2
        self.human_states = defaultdict(list)
        # 2.7 * 0.73
        self.human_radius = 1

        # robot
        self.max_speed = 1
        self.robot_radius = 0.3
        self.robot_states = list()

        # action space
        self.speed_samples = 3
        self.rotation_samples = 5
        self.actions = self._build_action_space()
        self.action_space = Discrete(self.speed_samples * self.rotation_samples + 1)

        # observation_space
        self.image_type = image_type
        self.observation_space = Box(low=0, high=255, shape=(84, 84, ImageInfo[image_type].channel_size))

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
        self.human_states = defaultdict(list)
        self.robot_states = list()

        if self.curriculum_learning:
            self.goal_distance = np.random.uniform(2, 4)
            self.goal_position = np.array((self.goal_distance, 0, -1))

        if self.robot_dynamics:
            self.client.reset()
        else:
            self.client.reset()
            self. _move(self.initial_position, 0)

        self._update_states()
        obs = self.compute_observation()

        return obs

    def step(self, action):
        import time
        pose = self.client.simGetVehiclePose()
        position = pose.position
        orientation = pose.orientation
        assert position.z_val == -1
        if self.robot_dynamics:
            car_controls = self._interpret_action(action)
            self.client.setCarControls(car_controls)
            if self.blocking:
                # pause for wall time self.time_step / self.clock_speed, which translates to game time self.time_step
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                while not self.client.simIsPause():
                    time.sleep(0.001)
        else:
            if isinstance(action, int):
                action_index = action
                action = self._interpret_action(action_index)

            if isinstance(action, ActionXY):
                x = position.x_val + action.vx * self.time_step
                y = position.y_val + action.vy * self.time_step
                yaw = np.arctan2(y, x)
            elif isinstance(action, ActionRot):
                _, _, yaw = airsim.to_eularian_angles(orientation)
                yaw += action.r
                x = position.x_val + action.v * np.cos(yaw) * self.time_step
                y = position.y_val + action.v * np.sin(yaw) * self.time_step
            else:
                raise NotImplementedError
            self._move((x, y, self.initial_position[2]), yaw)
            if self.blocking:
                self.client.simContinueForTime(self.time_step / self.clock_speed)
                while not self.client.simIsPause():
                    time.sleep(0.001)
        self.time += self.time_step
        self._update_states()

        past_pose = pose
        current_pose = self.client.simGetVehiclePose()
        if not (np.isclose(current_pose.position.x_val, x) and np.isclose(current_pose.position.y_val, y)):
            logging.debug('Different pose values between simGetVehiclePose and simSetVehiclePose!!!')
        dg = self._distance_to_goal(current_pose.position)
        collision_info = self.client.simGetCollisionInfo()

        if dg < self.robot_radius:
            reward = self.success_reward
            done = True
            info = 'Success'
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
            if self.reward_shaping:
                # early reward
                early_reward_range = self.goal_distance * self.early_reward_ratio
                heading_angle = np.arctan2(current_pose.position.y_val - past_pose.position.y_val,
                                           current_pose.position.x_val - past_pose.position.x_val)
                goal_angle = np.arctan2(self.goal_position[1] - past_pose.position.y_val,
                                        self.goal_position[0] - past_pose.position.x_val)
                if np.cos(heading_angle - goal_angle) > 0:
                    reward = max(0, min(1, (early_reward_range - dg) / (early_reward_range - self.robot_radius))) \
                             * self.success_reward
            done = False
            info = ''

        observation = self.compute_observation()

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def _move(self, pos, yaw):
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
            image = np.expand_dims(Image.fromarray(img2d).convert('L'), axis=2)
        else:
            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, image_type.channel_size)
            image = np.ascontiguousarray(image, dtype=np.uint8)

        # retrieve poses for both human and robot
        pose = self.client.simGetVehiclePose()
        r = self._distance_to_goal(pose.position)
        yaw = airsim.to_eularian_angles(pose.orientation)[2]
        phi = np.arctan2(self.goal_position[1] - pose.position.y_val, self.goal_position[0] - pose.position.x_val) - yaw
        goal = Goal(r, phi)
        logging.debug('Goal distance: {:.2f}, relative angle: {:.2f}'.format(r, np.rad2deg(phi)))

        observation = Observation(image, goal)

        return observation

    def _update_states(self):
        # retrieve all humans status
        for i in range(self.human_num):
            pose = self.client.simGetObjectPose('Human' + str(i))
            trials = 0
            while np.isnan(pose.position.x_val):
                pose = self.client.simGetObjectPose('Human' + str(i))
                trials += 1
                # logging.debug('Get NaN pose value. Try to call API again...')

                if trials >= 3:
                    logging.warning('Cannot get human status from client. Check human_num.')
                    break
            self.human_states[i].append(pose)
        self.robot_states.append(self.client.simGetVehiclePose())

    def compute_coordinate_observation(self, fov=True):
        # Todo: only consider humans in FOV
        human_states = []
        for i in range(self.human_num):
            if len(self.human_states[i]) == 1:
                vx = vy = 0
            else:
                vx = (self.human_states[i][-1].position.x_val - self.human_states[i][-2].position.x_val) / self.time_step
                vy = (self.human_states[i][-1].position.y_val - self.human_states[i][-2].position.y_val) / self.time_step
            px = self.human_states[i][-1].position.x_val
            py = self.human_states[i][-1].position.y_val
            human_state = ObservableState(px, py, vx, vy, self.human_radius)
            human_states.append(human_state)

        px = self.robot_states[-1].position.x_val
        py = self.robot_states[-1].position.y_val
        if len(self.robot_states) == 1:
            vx = vy = 0
        else:
            # TODO: use kinematics.linear_velocity?
            vx = self.robot_states[-1].position.x_val - self.robot_states[-2].position.x_val
            vy = self.robot_states[-1].position.y_val - self.robot_states[-2].position.y_val
        r  = self.robot_radius
        gx = self.goal_position[0]
        gy = self.goal_position[1]
        v_pref = 1
        _, _, theta = airsim.to_eularian_angles(self.robot_states[-1].orientation)
        robot_state = FullState(px, py, vx, vy, r, gx, gy, v_pref, theta)

        joint_state = JointState(robot_state, human_states)

        return joint_state

    def _interpret_action(self, action_index):
        assert isinstance(action_index, int)
        if self.robot_dynamics:
            car_controls = airsim.CarControls()
            car_controls.brake = 0
            car_controls.throttle = 1
            if action_index == 0:
                car_controls.throttle = 0
                car_controls.brake = 1
            elif action_index == 1:
                car_controls.steering = 0
            elif action_index == 2:
                car_controls.steering = 0.5
            elif action_index == 3:
                car_controls.steering = -0.5
            elif action_index == 4:
                car_controls.steering = 0.25
            else:
                car_controls.steering = -0.25
            return car_controls
        else:
            return self.actions[action_index]

    def _distance_to_goal(self, position):
        return norm(self.goal_position - vector2array(position))

    def _build_action_space(self):
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_speed for i in
                  range(self.speed_samples)]
        rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        actions = [ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            actions.append(ActionRot(speed, rotation))

        return actions


def vector2array(vector):
    assert type(vector) == airsim.Vector3r
    return np.array((vector.x_val, vector.y_val, vector.z_val))