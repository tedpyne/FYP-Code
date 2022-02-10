"""
Test Gym Wrapper of CARLO lanechange-V0 scenario with human controller (keyboard)

Goals:
    0: Reach point at end of the road in left lane
    1: Reach point at end of the road in right lane
"""

#!/usr/bin/env python3
import numpy as np
import gym_carlo
import gym
import time
import argparse
import matplotlib.pyplot as plt
from gym_carlo.envs.interactive_controllers import KeyboardController
from utils import *
from stable_baselines.common.env_checker import check_env

GOAL = 0

# SCENARIO_NAME = 'lanechange'
# SCENARIO_NAME = 'twocarlanechange'
# SCENARIO_NAME = 'threecarlanechange'
SCENARIO_NAME = 'intersection'
# SCENARIO_NAME = 'circularroad'
# SCENARIO_NAME = 'roundabout' # Not implemented

if __name__ == '__main__':
    # Sets up functionality to call script from cmd line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange, twocarlanechange, threecarlanechange, roundabout", default=SCENARIO_NAME)
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    # assert scenario_name in scenario_names, '--scenario argument is invalid!'
    
    
    env = gym.make(scenario_name + 'Scenario-v0', goal=GOAL)
    
    # Checks if env can be used to train
    # check_env(env)
    print('Environemnt passed Stable-Baselines check')
    
    env.seed(int(np.random.rand()*1e6))
    obs, done = env.reset(), False
    env.render()
    
    rewards = []
    interactive_policy = KeyboardController(env.world, steering_lims[scenario_name])
    while not done:
        t = time.time()
        a = [interactive_policy.steering, interactive_policy.throttle]
        obs ,reward, done, _ = env.step(a)
        rewards.append(reward)
        # print(reward)
        # print(env.ego.dist_to_centerline)
        print(env.ego.speed)
        if env.collision_exists == True:
            print("COLLISION OCCURED!")
            
        # if env.target_reached == True:
        #     print("SUCCESS!")
        
        env.render()
        env.write(scenario_name)
        while time.time() - t < env.dt/0.5: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
        
        
    env.close()
env.close()

rewards = rewards[:-1]
plt.plot([x for x in range(len(rewards))], rewards)
plt.title("Rewards")
plt.ylabel("Reward (r)")
plt.xlabel("Steps")