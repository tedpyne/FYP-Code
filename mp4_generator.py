# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:42:05 2021

@author: tedpy
"""
# TODO: Not Working
import gym
from gym.wrappers import Monitor


import numpy as np
import time
import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

# Build Environment
# SCENARIO_NAME = 'lanechange'
SCENARIO_NAME = 'twocarlanechange'
GOAL = 0

log_dir = "logs/" + SCENARIO_NAME + "/"

# Build env and saved VecNormalization statistics
env = gym.make(SCENARIO_NAME + 'Scenario-v0', goal=GOAL)
# env = Monitor(env, './video', force=True)
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: True,force=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(log_dir + 'vec_normalize.pkl', env)
env.training = False        #  do not update them at test time
env.norm_reward = False     # reward normalization is not needed at test time

# Load model
model = PPO2.load(log_dir + 'last_model')

obs = env.reset()
while True:
    t = time.time()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
        
    # while time.time() - t < env.dt/2: pass
    while time.time() - t < 0.2/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
    
    if done[0] == True:
        env.close()
        break 
env.close()


# import gym
# from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

# SCENARIO_NAME = 'twocarlanechange'
# GOAL = 0

# env_id = SCENARIO_NAME + 'Scenario-v0'
# video_folder = 'logs/videos/'
# video_length = 100

# env = DummyVecEnv([lambda: gym.make(env_id, goal=GOAL)])

# obs = env.reset()

# # Record the video starting at the first step
# env = VecVideoRecorder(env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix="PPO2-agent-{}".format(env_id))

# env.reset()
# for _ in range(video_length + 1):
#   action = [env.action_space.sample()]
#   obs, _, _, _ = env.step(action)
# # Save the video
# env.close()