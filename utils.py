"""
Contains kbject and functions utilities for training, testing, and results
analysis & plotting
"""
# import tensorflow as tf
import numpy as np
import os
from typing import Callable
from gym_carlo.envs.geometry import Point
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

scenario_names = ['intersection', 'circularroad', 'lanechange', 'twocarlanechange', 'threecarlanechange']
obs_sizes = {'intersection': 5,
             'circularroad': 4,
             'lanechange': 3,
             'twoCarLanechangeScenario': 3,
             'threeCarLanechangeScenario': 3
             }
goals = {'intersection': ['left','straight','right'],
         'circularroad': ['inner','outer'],
         'lanechange': ['left','right'],
         'twocarlanechange': ['left','right'],
         'threecarlanechange': ['left','right']
         }
# Steering Sensitivity [Left, Right]
steering_lims = {'intersection': [-0.5,0.5],
                 'circularroad': [-0.15,0.15],
                 'lanechange': [-0.15, 0.15],
                 'twocarlanechange': [-0.15, 0.15],
                 'threecarlanechange': [-0.15, 0.15]
                 }

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try: 
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def load_data(args):
    data_name = args.goal.lower()
    scenario_name = args.scenario.lower()
      
    assert scenario_name in goals.keys(), '--scenario argument is invalid!'
    data = {}
    if data_name == 'all':
        np_data = [np.load('data/' + scenario_name + '_' + dn + '.npy') for dn in goals[scenario_name]]
        u = np.vstack([np.ones((np_data[i].shape[0],1))*i for i in range(len(np_data))])
        np_data = np.vstack(np_data)
        data['u_train'] = np.array(u).astype('uint8').reshape(-1,1)
    else:
        assert data_name in goals[scenario_name], '--data argument is invalid!'
        np_data = np.load('data/' + scenario_name + '_' + data_name + '.npy')

    data['x_train'] = np_data[:,:-2].astype('float32')
    data['y_train'] = np_data[:,-2:].astype('float32') # control is always 2D: throttle and steering
    
    return data
    
   
def optimal_act_circularroad(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.6 + np.random.randn()*0.1
        
    # setting the steering is not fun. Let's practice some trigonometry
    r1 = 30.0 # inner building radius (not used rn)
    r2 = 39.2 # inner ring radius
    R = 32.3 # desired radius
    if d==1: R += 4.9
    Rp = np.sqrt(r2**2 - R**2) # distance between current "target" point and the current desired point
    theta = np.arctan2(env.ego.y - 60, env.ego.x - 60)
    target = Point(60 + R*np.cos(theta) + Rp*np.cos(3*np.pi/2-theta), 60 + R*np.sin(theta) - Rp*np.sin(3*np.pi/2-theta)) # this is pure magic (or I need to draw it to explain)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))
    if desired_heading >= h[hi]: steering = 0.15 + np.random.randn()*0.05
    else: steering = -0.15 + np.random.randn()*0.05
    return np.array([steering, throttle]).reshape(1,-1)
    
    
def optimal_act_lanechange(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.8 + np.random.randn()*0.1
        
    if d==0:
        target = Point(37.55, env.ego.y + env.ego.speed*3)
    elif d==1:
        target = Point(42.45, env.ego.y + env.ego.speed*3)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))
    if desired_heading >= h[hi]: steering = 0.15 + np.random.randn()*0.05
    else: steering = -0.15 + np.random.randn()*0.05
    return np.array([steering, throttle]).reshape(1,-1)

    return

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        
        self.is_tb_set = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        # TODO: Potential to log additional things to tensorboard here
        # # # Log additional tensor
        # if not self.is_tb_set:
        #     with self.model.graph.as_default():
        #         tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
        #         self.model.summary = tf.summary.merge_all()
        #     self.is_tb_set = True
        # # Log scalar value (here a random variable)
        # value = self.n_updates
        # summary = tf.Summary(value=[tf.Summary.Value(tag='n_updates', simple_value=value)])
        # self.locals['writer'].add_summary(summary, self.num_timesteps)
        
        return True
    
    
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_learning_curve(log_folder, title='Learning Curve'):
    """
    plots the smoothed learning curve (Rewards vs Timesteps)

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()