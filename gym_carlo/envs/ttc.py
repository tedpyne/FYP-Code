# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:15:14 2021

@author: tedpy
"""



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