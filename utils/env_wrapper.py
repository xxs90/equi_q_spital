import torch
import sys
sys.path.append('./')
sys.path.append('..')
import numpy as np
from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
from RLBench.rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
from RLBench.rlbench.environment import Environment
from RLBench.rlbench.observation_config import ObservationConfig
from RLBench.rlbench.tasks import PutItemInDrawer as tasks
import matplotlib.pyplot as plt
import utils

class EnvWrapper:
    def __init__(self, num_processes, simulator, env, env_config, planner_config):
        self.envs = Environment(action_mode=MoveArmThenGripper(
                                    arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
                                obs_config=ObservationConfig(),
                                headless=False,
                                robot_setup='panda')

    def reset(self):
        (states, in_hands, obs) = self.envs.reset()
        states = torch.tensor(states).float()
        in_hands = torch.tensor(in_hands).float()
        obs = torch.tensor(obs).float()
        return states, in_hands, obs

    def getNextAction(self):
        return torch.tensor(self.envs.getNextAction()).float()

    def step(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        (states_, in_hands_, obs_), rewards, dones = self.envs.step(actions, auto_reset)
        states_ = torch.tensor(states_).float()
        in_hands_ = torch.tensor(in_hands_).float()
        obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, in_hands_, obs_, rewards, dones

    def stepAsync(self, actions, auto_reset=False):
        actions = actions.cpu().numpy()
        self.envs.stepAsync(actions, auto_reset)

    def stepWait(self):
        (states_, in_hands_, obs_), rewards, dones = self.envs.stepWait()
        states_ = torch.tensor(states_).float()
        in_hands_ = torch.tensor(in_hands_).float()
        obs_ = torch.tensor(obs_).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        return states_, in_hands_, obs_, rewards, dones

    def getStepLeft(self):
        return torch.tensor(self.envs.getStepsLeft()).float()

    def reset_envs(self, env_nums):
        states, in_hands, obs = self.envs.reset_envs(env_nums)
        states = torch.tensor(states).float()
        in_hands = torch.tensor(in_hands).float()
        obs = torch.tensor(obs).float()
        return states, in_hands, obs

    def close(self):
        self.envs.close()

    def saveToFile(self, envs_save_path):
        return self.envs.saveToFile(envs_save_path)

    def getEnvGitHash(self):
        return self.envs.getEnvGitHash()

    def getEmptyInHand(self):
        return torch.tensor(self.envs.getEmptyInHand()).float()