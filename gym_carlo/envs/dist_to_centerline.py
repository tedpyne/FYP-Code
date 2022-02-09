# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:45:12 2021

@author: tedpy
"""

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
    
    if x_position < map_width / 2:
        return abs(x_position - left_lane_center)
    
    if x_position > map_width / 2:
        return abs(x_position - right_lane_center)
        
        