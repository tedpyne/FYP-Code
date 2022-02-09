# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:50:17 2021

@author: tedpy

Contains functions to calculate parameters of interest using informations
retrieved from scenario observations.

"""

from numpy import sqrt, arcsin, nan_to_num, pi

def distance_to_wall(x_position, map_width, building_width):
    
    if x_position < map_width / 2:
        return abs(x_position - building_width - 1)
    
    if x_position > map_width / 2:
        return abs(map_width - building_width - x_position - 1)
    

def distance_to_centerline(x_position, lane_pos, lane_width, map_width):
    """
    Computes the absoloute perpendicular distance of the vehicles center to the
            road centerline 
    
    Params:
        x_position: float: X-coordinate of the vehicle centre
        lane_pos: float: X-coordinate of the center lane
        lane_width: float: Width of each lane
        map_width float: Width of entire map
        
    Returns
        float: Abssoloute Perpendicular distance of vehicle center to road
                centerline
        
    """
    left_lane_center = lane_pos - (lane_width / 2)
    right_lane_center = lane_pos + (lane_width / 2)
    
    # Centerline left lane
    if x_position <= map_width / 2:
        return abs(x_position - left_lane_center)
    
    # Centerline right lane
    if x_position > map_width / 2:
        return  abs(x_position - right_lane_center)
    
def distance_to_centerline_intersection(self, lane_pos, lane_width, map_width, building_height):
    """
    Computes the absoloute perpendicular distance of the vehicles center to the
            road centerline 
    
    Params:
        x_position: float: X-coordinate of the vehicle centre
        lane_pos: float: X-coordinate of the center lane
        lane_width: float: Width of each lane
        map_width float: Width of entire map
        
    Returns
        float: Abssoloute Perpendicular distance of vehicle center to road
                centerline
        
    """
    x_position = self.ego.center.x
    y_position = self.ego.center.y
    left_lane_center = lane_pos - (lane_width / 2)
    
    # Lower striaght
    # if self.ego.y < building_height + self.intersection_y - 5:
    if self.ego.y < self.intersection_y - lane_width:
        return abs(x_position - left_lane_center)
    
    # After intersection
    elif self.ego_approaching_intersection == False:
        # Straight
        if self.active_goal == 1:
            return abs(x_position - left_lane_center)
        
        # Left Turn
        elif self.active_goal == 0:
            return abs(y_position - (self.intersection_y - (lane_width / 2)))
        
        # Right turn
        elif self.active_goal == 2:
            return abs(y_position - (self.intersection_y + (lane_width / 2)))
    
    # In intersection
    else:
        # print("Intersection")
        # Straight
        if self.active_goal == 1:
            return abs(x_position - left_lane_center)
        
        # Left Turn
        elif self.active_goal == 0:
            # Given two points on circle arc
            x_lower = (map_width/2) - (lane_width/2)
            y_lower = self.intersection_y - lane_width
            x_upper = (map_width/2) - lane_width
            y_upper = self.intersection_y - (lane_width/2)
            
            # Circle center and radius
            x_center = x_upper
            y_center = y_lower
            radius = y_upper - y_center
            
            # Distance from ego to arc
            return abs(sqrt(((self.ego.center.x - x_center)**2) + (self.ego.center.y - y_center)**2) - radius)
        
        # Right turn
        elif self.active_goal == 2:
            # Given two points on circle arc
            x_lower = (map_width/2) - (lane_width/2)
            y_lower = self.intersection_y - lane_width
            x_upper = (map_width/2) + lane_width
            y_upper = self.intersection_y + (lane_width/2)
            
            # Circle center and radius
            x_center = x_upper
            y_center = y_lower
            radius = y_upper - y_center
            
            # Distance from ego to arc
            return abs(sqrt(((self.ego.center.x - x_center)**2) + (self.ego.center.y - y_center)**2) - radius)
        
            
            
    

def tlc(x_position, speed, uncertainty, lane_pos, lane_width, car_width, map_width):
    # TODO:Add consideration for car size. Currently tlc calculated for center point
    # of vehicle
    """
    Computes the Time-to-Line-Crossing for a given vehicle.
    
    Time-to-Line-Cross algorithm based off
    "Satisficing Curve Negotiation: Explaining Drivers’ Situated Lateral
    Position Variability" 
    by Erwin R. Boer
    
    Params:
        x_position: float: X-coordinate of the vehicle centre
        speed: float: Speed of vehicle
        Uncertainty: float: Measure of steering uncertainty
        lane_pos: float: X-coordinate of the center lane
        lane_width: float: Width of each lane
        car_width: flaot: Width of all vehicles
        map_width float: Width of entire map.
        
    Returns
        tuple (tlc_neg, tlc_pos)
    """
    
    # Geometry
    #lane_pos = lane_pos - car_width # Correction to give effective lane width
    left_lane_center = lane_pos - (lane_width / 2)
    right_lane_center = lane_pos + (lane_width / 2)
    
    if x_position < map_width / 2:
        delta = x_position - left_lane_center
    if x_position > map_width / 2:
        delta = x_position - right_lane_center
        
    R_lambda = 1 / uncertainty
    
    # TLC Straight Positive (Right Side)
    eta_pos = (lane_width / 2) - delta
    D_pos = sqrt((2 * R_lambda * eta_pos) - eta_pos**2)
    phi_pos = arcsin(D_pos / R_lambda)
    D_curved_pos = R_lambda * phi_pos
    tlc_pos = D_curved_pos / speed
        
    # TLC Straight Negative   (Left Side)  
    eta_neg = (lane_width / 2) + delta
    D_neg = sqrt((2 * R_lambda * eta_neg) - eta_neg**2)
    phi_neg = arcsin(D_neg / R_lambda)
    D_curved_neg = R_lambda * phi_neg
    tlc_neg = D_curved_neg / speed
    
    tlc = [tlc_neg, tlc_pos]
    tlc = nan_to_num(tlc) # Fix for NaN's when off road
    
    return tlc

def ttc(ego_center_x, ego_center_y, ego_speed, ado_center_x, ado_center_y, ado_speed):
    """
    Simple Time-to-Collision algorithms. Does not account for
    non-linear trajectories or accelerations.
    
    Params:
        ego_center_x: float:
        ego_center_y: float:
        ego_speed: float:
        ado_center_x: float:
        ado_center_y: float:
        ado_speed: float:
    
    Returns:
        ttc : float:
    """
    # TODO: Reports None after ovetake but this is not neccessarily true.
    # TODO: Only works when ego is behind ado. What about side on or diagonal collisions
    # TODO: Improve implementation of collision. Differentiate between
    
    if ego_center_y < ado_center_y:
        if ego_speed > ado_speed:
            # Size of Car defined in agents.py. Length = 4, Width = 2.
            top_left_ego = ego_center_x - 1
            top_left_other = ado_center_x - 1
            top_right_other = ado_center_x + 1
            if top_left_ego >= top_left_other-2 and top_left_ego <= top_right_other+2:
                d = abs(ego_center_y - ado_center_y)
                return (d-4)/(ego_speed - ado_speed)
            
def reward(self, LANE_WIDTH):
    """
    Computes reward score (r)
    
    Params:
        collision_exists: bool
        target_reached: bool
        
    Returns:
        reward
    """
    
    if self.collision_exists:
        return -500
    
    if self.target_reached:
        return 500
    
    if self.active_goal < len(self.targets):
        # Objective signal
        obj_signal = 20. / self.targets[self.active_goal].distanceTo(self.ego)
        
        # Time signal
        time_signal = 0.02 * self.world.t
        
        # Velocity Signal
        if self.ego.velocity.y <= self.init_ego.speed_limit:
            vel_signal = 0.5 * (self.init_ego.speed_limit - self.ego.velocity.y)
        else:
            vel_signal = 0.75 * (self.ego.velocity.y - self.init_ego.speed_limit)
         
        # Centerline Signal
        if self.ego.dist_to_centerline <= LANE_WIDTH / 2:
            centerline_signal = 0.5 * self.ego.dist_to_centerline
        else:
            centerline_signal = 2 * self.ego.dist_to_centerline

        
        # Smooth driving signal
        angular_v_signal = 0.3 * abs(self.ego.angular_acceleration)
        # angular_v_signal = 0
        
        reward = obj_signal - time_signal - vel_signal - centerline_signal - angular_v_signal

        return reward
    
def reward_intersection(self, LANE_WIDTH):
    
    # Termination rewards
    if self.collision_exists:
        return -500
    if self.target_reached:
        return 500
    
    # Driven past target
    if self.active_goal == 0:
        if self.ego.center.y > 65 or self.ego.center.x > 42:
            return -50
    elif self.active_goal == 1:
        if self.ego.center.x < 33 or self.ego.center.x > 42:
            return -50
    elif self.active_goal == 2:
        if self.ego.center.y > 70 or self.ego.center.x < 33:
            return -50
        
    # Proximity to pedestrians
    if self.ped_1.distanceTo(self.ego) < 5:
        return -50
    elif self.ped_2.distanceTo(self.ego) < 5:
        return -50
    
    if self.active_goal < len(self.targets):
        
        # Objective signal
        obj_signal = 10 - (0.01*self.targets[self.active_goal].distanceTo(self.ego))**4
        # print("Objective Signal: {}".format(obj_signal))
        
        # Time signal
        time_signal = 0.1 * self.world.t
        # print("Time Signal: {}".format(time_signal))
        
        vel_signal = 10. / (1 + abs(self.init_ego.speed_limit - self.ego.speed))
        if self.ego.speed < 5:
            vel_signal = vel_signal - 20 
        # print("Velocity Signal: {}".format(vel_signal))
        
        # Centerline Signal
        centerline_signal = 5./ (1 + self.ego.dist_to_centerline)
        if self.ego.dist_to_centerline > 2:
            centerline_signal = centerline_signal - 5
        # print("Centerline Signal: {}".format(centerline_signal))
        
        # Smooth driving signal
        # angular_v_signal = 0.3 * abs(self.ego.angular_acceleration)
        angular_v_signal = 0
        
        reward = obj_signal - time_signal + vel_signal + centerline_signal - angular_v_signal
        return reward
    
