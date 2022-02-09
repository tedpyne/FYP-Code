# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:48:54 2021
"""
# TODO:Add consideration for car size. Currently tlc calculated for center point
# of vehicle


from numpy import sqrt, arcsin

def tlc(x_position, speed, uncertainty, lane_pos, lane_width, car_width, map_width):
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
    
    return tlc_neg, tlc_pos