# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:33:01 2021

@author: tedpy
"""

#!/usr/bin/env python3
import time
import argparse
import numpy as np

import gym_carlo
import gym
from gym_carlo.envs.interactive_controllers import KeyboardController
from utils import *

from stable_baselines.bench import Monitor
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.gail import generate_expert_traj

SCENARIO_NAME = 'lanechange'
GOAL = 0

# Dirs
log_dir = "logs/" + SCENARIO_NAME + "/"
save_dir = "pretrain_datasets/" + SCENARIO_NAME + "/"

# Build env
env = gym.make(SCENARIO_NAME + 'Scenario-v0', goal=0)
env.seed(int(np.random.rand()*1e6))
check_env(env)
print('Environemnt passed Stable-Baselines check')

# # Build Environment
# env = gym.make(SCENARIO_NAME + 'Scenario-v0', goal=GOAL)
# env = DummyVecEnv([lambda: env])
# # Much better performance with obs, rewards, and actions normalized
# env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                    clip_obs=10.)

# Expert Agent 
def dummy_expert(_obs):
    
    return env.action_space.sample()

def keyboard_expert(_obs):
    obs, done = env.reset(), False
    env.render()
    interactive_policy = KeyboardController(env.world, steering_lims[SCENARIO_NAME])
    while not done:
        t = time.time()
        a = [interactive_policy.steering, interactive_policy.throttle]
        obs ,reward, done, _ = env.step(a)
        if env.collision_exists == True:
            print("COLLISION OCCURED!")
            
        if env.target_reached == True:
            print("SUCCESS!")
        
        env.render()
        #env.write('test')
        while time.time() - t < env.dt/1: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
    # env.close()    
        return env.action_space.sample()

# Generate dataset
# model = PPO2(MlpPolicy, env, verbose=1)
generate_expert_traj(dummy_expert, save_dir+'norm_dummy_expert', env, n_episodes=10)

dataset = np.load(save_dir + 'dummy_expert.npz')
lst = dataset.files
for item in lst:
    print(item)
    print(dataset[item])
    
norm_dataset = np.load(save_dir + 'norm_dummy_expert.npz')
lst = norm_dataset.files
for item in lst:
    print(item)
    print(norm_dataset[item])
