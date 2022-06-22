import torch
import sys
sys.path.append('./')
sys.path.append('..')
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutItemInDrawer as tasks
import matplotlib.pyplot as plt
import utils
#
# class EnvWrapper:
#     def __init__(self, num_processes, simulator, env, env_config, planner_config):
#         self.envs = Environment(action_mode=MoveArmThenGripper(
#                                     arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
#                                 obs_config=ObservationConfig(),
#                                 headless=False,
#                                 robot_setup='panda')
#
#     def reset(self):
#         (states, in_hands, obs) = self.envs.reset()
#         states = torch.tensor(states).float()
#         in_hands = torch.tensor(in_hands).float()
#         obs = torch.tensor(obs).float()
#         return states, in_hands, obs
#
#     def getNextAction(self):
#         return torch.tensor(self.envs.getNextAction()).float()
#
#     def step(self, actions, auto_reset=False):
#         actions = actions.cpu().numpy()
#         (states_, in_hands_, obs_), rewards, dones = self.envs.step(actions, auto_reset)
#         states_ = torch.tensor(states_).float()
#         in_hands_ = torch.tensor(in_hands_).float()
#         obs_ = torch.tensor(obs_).float()
#         rewards = torch.tensor(rewards).float()
#         dones = torch.tensor(dones).float()
#         return states_, in_hands_, obs_, rewards, dones
#
#     def stepAsync(self, actions, auto_reset=False):
#         actions = actions.cpu().numpy()
#         self.envs.stepAsync(actions, auto_reset)
#
#     def stepWait(self):
#         (states_, in_hands_, obs_), rewards, dones = self.envs.stepWait()
#         states_ = torch.tensor(states_).float()
#         in_hands_ = torch.tensor(in_hands_).float()
#         obs_ = torch.tensor(obs_).float()
#         rewards = torch.tensor(rewards).float()
#         dones = torch.tensor(dones).float()
#         return states_, in_hands_, obs_, rewards, dones
#
#     def getStepLeft(self):
#         return torch.tensor(self.envs.getStepsLeft()).float()
#
#     def reset_envs(self, env_nums):
#         states, in_hands, obs = self.envs.reset_envs(env_nums)
#         states = torch.tensor(states).float()
#         in_hands = torch.tensor(in_hands).float()
#         obs = torch.tensor(obs).float()
#         return states, in_hands, obs
#
#     def close(self):
#         self.envs.close()
#
#     def saveToFile(self, envs_save_path):
#         return self.envs.saveToFile(envs_save_path)
#
#     def getEnvGitHash(self):
#         return self.envs.getEnvGitHash()
#
#     def getEmptyInHand(self):
#         return torch.tensor(self.envs.getEmptyInHand()).float()


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        arm = [-4.0, -2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        print(arm)

        # arm = np.random.normal(1.0, 1.0, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False,
    robot_setup='panda')
env.launch()

task = env.get_task(tasks)
agent = Agent(env.action_shape)

# demos = task.get_demos(2, live_demos=True)

# rc.get_joint_positions(agent)

training_steps = 2
episode_length = 1
# obs = None
x1 = 0 #-30.0 (lr)
x2 = 0.3 #30.0
y1 = 0#-30.0
y2 = 0.0#30.0
z1 = 1.2
z2 = 1.4
q_i = 0.0
q_j = 1.0
q_k = 0.0
q = 0.0
# action = [[x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0]]
action = [x2, y2, z2, q_i, q_j, q_k, q, 1.0]
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    # action = agent.act(obs)
    # print(action)
    obs, reward, terminate = task.step(action)

    # fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    # ax[0].imshow(obs.front_depth)
    # ax[0].set_title('front')
    # ax[1].imshow(obs.overhead_depth)
    # ax[1].set_title('overhead')
    # ax[2].imshow(obs.wrist_depth)
    # ax[2].set_title('wrist')
    # ax[3].imshow(obs.left_shoulder_depth)
    # ax[3].set_title('left_shoulder')
    # ax[4].imshow(obs.right_shoulder_depth)
    # ax[4].set_title('right_shoulder')
    # plt.show()

    cloud = utils.combinePointClouds(obs)
    depth_img = utils.getProjectImg(cloud, 1.0, 128, (0.35, 0.0, 1.2))
    plt.imshow(depth_img)
    plt.colorbar()
    plt.show()

print('Done')
env.shutdown()
