# To launch Tensorboard during training:
    # 1. Navigate to tensorboard log dir in anaconda prompt (AVRIL venv)
    # 2. python -m tensorboard.main --logdir=PPO2_<NUM>/ --host=127.0.0.1
    # 3. http://localhost:6006/

import numpy as np
import os
import gym
import matplotlib.pyplot as plt

import utils

import stable_baselines
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import TRPO
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.results_plotter import ts2xy

# Parameters
SCENARIO_NAME = 'lanechange'
# SCENARIO_NAME = 'twocarlanechange'
# SCENARIO_NAME = 'threecarlanechange'
GOAL = 0

# dirs
log_dir = "logs/TRPO/" + SCENARIO_NAME + "/"
os.makedirs(log_dir, exist_ok=True)

# Build Environment
env = gym.make(SCENARIO_NAME + 'Scenario-v0', goal=GOAL)
check_env(env)
print('Environemnt passed Stable-Baselines check')

env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

# Build & Train model
callback = utils.SaveOnBestTrainingRewardCallback(check_freq=50000,
                                                  log_dir=log_dir)
time_steps = 100_000

model = TRPO(MlpPolicy, env, verbose=1,
              tensorboard_log=log_dir)

model.learn(total_timesteps=time_steps, callback=callback)

model.save(log_dir + 'last_model')
env.save(log_dir + 'vec_normalize.pkl')

# Results Plots
results_df = stable_baselines.bench.monitor.load_results(log_dir)
results_df['index'] = ts2xy(results_df, 'timesteps')[0]
results_df = results_df.rename(columns={'index' : 'Timesteps',
                                        'r':'Episode Reward (Mean)',
                                        'l':'Episode Length',
                                        't':'Episode Runtime'})

# Monitor Results
results_df.plot(x='Timesteps', y=['Episode Reward (Mean)', 'Episode Length'],
                title=SCENARIO_NAME + 'Training Monitor (TRPO)', subplots=True)

# Timesteps vs Episode Reward (Points)
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS,
                             "lanechange (TRPO)")

plt.show()
# Timesteps vs Episode Reward (Time Series)
utils.plot_learning_curve(log_dir)


