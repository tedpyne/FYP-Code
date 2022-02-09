# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:55:35 2021

@author: tedpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

# Build Environment
# SCENARIO_NAME = 'lanechange'
# SCENARIO_NAME = 'twocarlanechange'
# SCENARIO_NAME = 'threecarlanechange'
SCENARIO_NAME = 'intersection'

GOAL = 0

log_dir = "logs/PPO/" + SCENARIO_NAME + "/"

# Build env and saved VecNormalization statistics
env = gym.make(SCENARIO_NAME + 'Scenario-v0', goal=GOAL)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(log_dir + 'vec_normalize.pkl', env)
env.training = False        #  do not update them at test time
env.norm_reward = False     # reward normalization is not needed at test time

# Load model
model = PPO2.load(log_dir + 'last_model')

# Initialise dict for performance tracking
keys = ["Steering (Input)", "Throttle (Input)","Speed (m/s)", "Vx (m/s)", "Vy (m/s)",
        "Angular Velocity (1/s)", "Angular Acceleration (1/s^2)",
        "Heading (m/s)", "Acceleration (m/s^2)", "Jerk (m/s^3)",
        "Distance to Centerline (m)", "Distance to Target (m)",
        "Distance to Wall (m)", "Time-to-Cross (Left) (s)",
        "Time-to-Cross (Right) (s)", "Reward"]
ego_attributes = dict((key, []) for key in keys)

# keys = ["Target", "Time", "Vy", "Centerline", "Jerk", "Angular_V"]

obs = env.reset()
while True:
    t = time.time()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    
    
    env.render()

    # Track ego vehicle attributes
    ego = env.get_attr('ego')[0]
    ego_attributes[keys[0]].append(action[0][0])
    ego_attributes[keys[1]].append(action[0][1])
    ego_attributes[keys[2]].append(ego.speed)
    ego_attributes[keys[3]].append(ego.velocity.x)
    ego_attributes[keys[4]].append(ego.velocity.y)
    ego_attributes[keys[5]].append(ego.angular_velocity)
    ego_attributes[keys[6]].append(ego.angular_acceleration)
    ego_attributes[keys[7]].append(ego.heading)
    ego_attributes[keys[8]].append(ego.acceleration)
    ego_attributes[keys[9]].append(ego.jerk)
    ego_attributes[keys[10]].append(ego.dist_to_centerline)
    ego_attributes[keys[11]].append(ego.dist_to_target)
    ego_attributes[keys[12]].append(ego.dist_to_wall)
    ego_attributes[keys[13]].append(ego.tlc[0])
    ego_attributes[keys[14]].append(ego.tlc[1])
    ego_attributes[keys[15]].append(rewards[0])
    
    print(info)
        
    # while time.time() - t < env.dt/2: pass
    while time.time() - t < 0.2/2: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
        
    if done[0] == True:
        env.close()
        break 
env.close()


# # Plot Results
df = pd.DataFrame.from_dict(ego_attributes)
df = df[:-1] # Remove last valeus due to env.reset() 
df.plot(subplots=True, title=keys, layout=(4,4), legend=False, figsize=(10,10))

# df = pd.DataFrame.from_dict(info)
# for i in df.columns:
#     plt.figure()
#     plt.plot(df[i])