# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:34:23 2022

@author: tedpy
"""

def reward(self):
    """
    Computes reward score (r)
    
    Params:
        collision_exists: bool
        target_reached: bool
        
    Returns:
        reward
    """
    
    if self.collision_exists:
        return -100
    
    if self.target_reached:
        return 100
    
    if self.active_goal < len(self.targets):
        # Get to objective signal
        reward = 10. / self.targets[self.active_goal].distanceTo(self.ego)

        # Time signal
        reward = reward - (0.01 * self.world.t)
        
        # Velocity Signal
        reward = reward - (0.01 * (self.init_ego.max_speed - self.ego.velocity.y))

        # Centerline signal
        reward = reward - (0.02 * self.ego.dist_to_centerline)
        
        # Smooth Driving Signal
        reward = reward - (0.005 * self.ego.jerk)
        reward = reward - (0.005 * self.ego.angular_acceleration)

        
        return reward