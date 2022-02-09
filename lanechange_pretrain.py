# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:07:51 2021

@author: tedpy
"""
import gym
import numpy as np

from stable_baselines.gail import ExpertDataset
from stable_baselines.common import make_vec_env
from stable_baselines.bench import Monitor
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

SCENARIO_NAME = 'lanechange'

data_dir = "pretrain_datasets/" + SCENARIO_NAME + "/"
log_dir = "logs/" + SCENARIO_NAME + "/"

dataset = ExpertDataset(expert_path=data_dir+'dummy_expert.npz',
                        traj_limitation=1)

env = make_vec_env(SCENARIO_NAME + 'Scenario-v0', n_envs=1,
                    seed=int(np.random.rand()*1e6))

model = PPO2(MlpPolicy,
             env,
             verbose=1)

model.pretrain(dataset)