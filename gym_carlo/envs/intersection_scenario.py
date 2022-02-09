import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import random

from .world import World
from .agents import Car, RectangleBuilding, Pedestrian, Painting
from .geometry import Point
from .graphics import Text, Point as pnt # very unfortunate indeed

from .obs_utils import *


MAP_WIDTH = 80
MAP_HEIGHT = 120
# LANE_WIDTH = 4.4
LANE_WIDTH = 5
SIDEWALK_WIDTH = 2.0
LANE_MARKER_HEIGHT = 3.8
LANE_MARKER_WIDTH = 0.5
BUILDING_WIDTH = (MAP_WIDTH - 2*SIDEWALK_WIDTH - 2*LANE_WIDTH - LANE_MARKER_WIDTH) / 2.
TOP_BUILDING_HEIGHT = MAP_HEIGHT - (LANE_MARKER_WIDTH/2. + LANE_WIDTH + SIDEWALK_WIDTH) # intersection_y will be subtracted
BOTTOM_BUILDING_HEIGHT = -LANE_MARKER_WIDTH/2. - LANE_WIDTH - SIDEWALK_WIDTH # intersection_y will be added

PPM = 5 # pixels per meter

class IntersectionScenario(gym.Env):
    def __init__(self, goal):
        assert 0 <= goal <= 3, 'Undefined goal'
        
        self.num_envs = 1
    
        self.seed(0) # just in case we forget seeding
        
        self.active_goal = goal
        
        self.init_ego = Car(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., 0), heading = np.pi/2)
        self.init_ego.velocity = Point(1., 0.)
        self.init_ego.min_speed = 0.
        self.init_ego.max_speed = 30.
        self.init_ego.speed_limit = 8.34    # Approx 30 Km/h (typical urban speed limit)
        
        self.ped_min_speed = 0.5
        self.ped_max_speed = 3.
        
        # [center_x, center_y, velocity_y, heading,
        #  centerline, target_x,, target_y, intersection_y,
        #  ped_1_center_x, ped_1_center_y, ped_1_velocity_y, ped_1_heading,
        #  ped_2_center_x, ped_2_center_y, ped_2_velocity_y, ped_2_heading]
        self.observation_space = Box(low=np.array([0,
                                                   0,
                                                   self.init_ego.min_speed,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                    0,
                                                    0,
                                                    self.ped_min_speed,
                                                    0,
                                                    0,
                                                    0,
                                                    self.ped_min_speed,
                                                    0
                                                   ]),
                                     high=np.array([MAP_WIDTH,
                                                    MAP_HEIGHT,
                                                    self.init_ego.max_speed,
                                                    2*np.pi,
                                                    MAP_WIDTH/2, 
                                                    MAP_WIDTH,
                                                    MAP_HEIGHT,
                                                    MAP_HEIGHT,
                                                    MAP_WIDTH,
                                                    MAP_HEIGHT,
                                                    self.init_ego.max_speed,
                                                    2*np.pi,
                                                    MAP_WIDTH,
                                                    MAP_HEIGHT,
                                                    self.init_ego.max_speed,
                                                    2*np.pi
                                                    ]),
                                     shape=(16,),
                                     dtype=np.float32)
        # Continuous 
        self.action_space = Box(low=np.float32(np.array([-0.45,-1])),
                                high=np.float32(np.array([0.45,1])))
        
        self.dt = 0.1
        self.T = 50
        
        self.reset()
        
    def reset(self):
        # Randomly choose a target for training
        # self.active_goal = random.choice([0,1,2])
        
        self.world = World(self.dt, width = MAP_WIDTH, height = MAP_HEIGHT, ppm = PPM)
        
        self.intersection_y = MAP_HEIGHT/2
        # self.intersection_y = random.uniform(0,1)*MAP_HEIGHT/2. + MAP_HEIGHT/4.
           
        self.ego = self.init_ego.copy()
        self.ego.center += Point(self.np_random.rand() - 5.5, self.np_random.rand()*(self.intersection_y-MAP_HEIGHT/10.))
        self.ego.heading += np.random.randn()*0.05
        self.ego.velocity = Point(0, self.np_random.rand()*5)
        
        self.ped_1 = Pedestrian(Point(BUILDING_WIDTH + SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2.), heading=0)
        self.ped_1.center = Point(BUILDING_WIDTH + SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2. + .6)
        self.ped_1.heading = 0
        self.ped_1.velocity = Point(random.uniform(0.5, 3), 0)
        
        self.ped_2 = Pedestrian(Point(MAP_WIDTH - BUILDING_WIDTH - SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2.), heading=0)
        self.ped_2.center = Point(MAP_WIDTH - BUILDING_WIDTH - SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2. - 0.4)
        self.ped_2.heading = np.pi
        self.ped_2.velocity = Point(random.uniform(0.5, 3), 0)
       
        self.targets = []
        self.targets.append(Point(0., self.intersection_y - LANE_WIDTH/2. - LANE_MARKER_WIDTH/2.))          # Left
        self.targets.append(Point(MAP_WIDTH/2. - LANE_WIDTH/2. - LANE_MARKER_WIDTH/2., MAP_HEIGHT))         # Straight
        self.targets.append(Point(MAP_WIDTH, self.intersection_y + LANE_WIDTH/2. + LANE_MARKER_WIDTH/2.))   # Right
        self.world.add(Painting(self.targets[self.active_goal], Point(2,2),'yellow'))
        
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))

        # lane markers on the bottom road
        for y in np.arange(LANE_MARKER_HEIGHT/2., self.intersection_y - LANE_MARKER_WIDTH/2 - LANE_WIDTH - LANE_MARKER_HEIGHT/2, 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        # lane markers on the right road
        for x in np.arange(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2 + LANE_WIDTH + LANE_MARKER_HEIGHT/2, MAP_WIDTH - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(x, self.intersection_y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white', heading = np.pi/2))
        # lane markers on the top road
        for y in np.arange(self.intersection_y + LANE_MARKER_WIDTH/2 + LANE_WIDTH + LANE_MARKER_HEIGHT/2, MAP_HEIGHT - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        # lane markers on the left road
        for x in np.arange(LANE_MARKER_HEIGHT/2, MAP_WIDTH/2. - LANE_MARKER_WIDTH/2 - LANE_WIDTH - LANE_MARKER_HEIGHT/2, 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(x, self.intersection_y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white', heading = np.pi/2))

        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y)))
        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y)))
        
        # self.world.add(Pedestrian(Point(BUILDING_WIDTH + SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2.), 0))
        # self.world.add(Pedestrian(Point(MAP_WIDTH - BUILDING_WIDTH - SIDEWALK_WIDTH/2.,(BOTTOM_BUILDING_HEIGHT+self.intersection_y) + SIDEWALK_WIDTH/2.), 0))
        self.world.add(self.ego)
        self.world.add(self.ped_1)
        self.world.add(self.ped_2)
        # self.world.add(Painting(Point(MAP_WIDTH/2, self.intersection_y), Point(1,1),'red'))
        
        self.ego.dist_to_centerline = distance_to_centerline_intersection(self, MAP_WIDTH/2., LANE_WIDTH, MAP_WIDTH, BOTTOM_BUILDING_HEIGHT)
        # self.ego.dist_to_hard_shoulder = distance_to_centerline(self.ego.center.x, MAP_WIDTH/2., LANE_WIDTH, MAP_WIDTH)
        # self.ego.tlc = tlc(self.ego.center.x, self.ego.speed, 0.006, MAP_WIDTH/2., LANE_WIDTH, self.ego.size.x, MAP_WIDTH)
        self.ego.dist_to_target =  self.targets[self.active_goal].distanceTo(self.ego)
        self.ego.dist_to_wall = distance_to_wall(self.ego.center.x, MAP_WIDTH, BUILDING_WIDTH)
        
        return self._get_obs()
        
    def close(self):
        self.world.close()
        
    
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def target_reached(self):
        if self.active_goal < len(self.targets):
            return self.targets[self.active_goal].distanceTo(self.ego) < 1.
        return np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))]) < 1.
    
    @property
    def collision_exists(self):
        return self.world.collision_exists()
        
    @property
    def ego_approaching_intersection(self):
        return MAP_HEIGHT - (TOP_BUILDING_HEIGHT - self.intersection_y) - SIDEWALK_WIDTH > self.ego.y > BOTTOM_BUILDING_HEIGHT + self.intersection_y - 4 and \
                BUILDING_WIDTH + SIDEWALK_WIDTH < self.ego.x < MAP_WIDTH - BUILDING_WIDTH - SIDEWALK_WIDTH
                
    # @property
    # def ego_approaching_intersection(self):
    #     return MAP_HEIGHT - TOP_BUILDING_HEIGHT - SIDEWALK_WIDTH > self.ego.y > BOTTOM_BUILDING_HEIGHT + SIDEWALK_WIDTH and \
    #             BUILDING_WIDTH + SIDEWALK_WIDTH < self.ego.x < MAP_WIDTH - BUILDING_WIDTH - SIDEWALK_WIDTH
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ego.set_control(action[0],action[1])
        self.world.tick()
        
        return self._get_obs(), self._get_reward(), self.collision_exists or self.target_reached or self.ego.y > MAP_HEIGHT or self.ego.x > MAP_WIDTH or self.ego.x < 0 or self.world.t >= self.T, {}
    
    def _get_reward(self):
            return reward_intersection(self, LANE_WIDTH)
        
    def _get_obs(self):
        self.ego.dist_to_centerline = distance_to_centerline_intersection(self, MAP_WIDTH/2., LANE_WIDTH, MAP_WIDTH, BOTTOM_BUILDING_HEIGHT)
        self.ego.tlc = tlc(self.ego.center.x, self.ego.speed, 0.006, MAP_WIDTH/2., LANE_WIDTH, self.ego.size.x, MAP_WIDTH)
        self.ego.dist_to_target =  self.targets[self.active_goal].distanceTo(self.ego)
        self.ego.dist_to_wall = distance_to_wall(self.ego.center.x, MAP_WIDTH, BUILDING_WIDTH)
        
        # [center_x, center_y, velocity_y, heading,
        #  centerline, target_x,, target_y, intersection_y,
        #  ped_1_center_x, ped_1_center_y, ped_1_velocity_y, ped_1_heading,
        #  ped_2_center_x, ped_2_center_y, ped_2_velocity_y, ped_2_heading]
        return np.float32(np.array([self.ego.center.x,
                         self.ego.center.y,
                         self.ego.velocity.y,
                         self.ego.heading,
                         self.ego.dist_to_centerline,
                         self.targets[self.active_goal].x,
                         self.targets[self.active_goal].y,
                         self.intersection_y,
                          self.ped_1.center.x,
                          self.ped_1.center.y,
                          self.ped_1.velocity.y,
                          self.ped_1.heading,
                          self.ped_2.center.x,
                          self.ped_2.center.y,
                          self.ped_2.velocity.y,
                          self.ped_2.heading
                         ]))
        
    def render(self, mode='rgb'):
        self.world.render()
        
    def write(self, text): # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH - BUILDING_WIDTH+2), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)