import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import random

from .world import World
from .agents import Car, RectangleBuilding, Pedestrian, Painting
from .geometry import Point
from .graphics import Text, Point as pnt


from .obs_utils import tlc, distance_to_centerline, distance_to_wall, reward

MAP_WIDTH = 80
MAP_HEIGHT = 120
LANE_WIDTH = 4.4
SIDEWALK_WIDTH = 2.0
LANE_MARKER_HEIGHT = 3.8
LANE_MARKER_WIDTH = 0.5
BUILDING_WIDTH = (MAP_WIDTH - 2*SIDEWALK_WIDTH - 2*LANE_WIDTH - LANE_MARKER_WIDTH) / 2.

PPM = 5 # pixels per meter

class twoCarLanechangeScenario(gym.Env):
    def __init__(self, goal):
        assert 0 <= goal <= 3, 'Undefined goal'
    
        self.seed(0) # just in case we forget seeding
        
        self.num_envs = 1
        
        self.active_goal = goal
        
        self.init_ego = Car(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., 0), np.pi/2, 'red')
        self.init_ego.velocity = Point(1., 0.)
        self.init_ego.min_speed = 0.
        self.init_ego.max_speed = 15.
        self.init_ego.speed_limit = 8.34    # Approx 30 Km/h (typical urban speed limit)
        
        
        self.init_c2 = Car(Point(40,60), np.pi/2 , 'blue')
        self.init_c2.velocity = Point(1., 0.)
        self.init_c2.min_speed = 0.
        self.init_c2.max_speed = 10.
        
        # [center_x_ego, center_y_ego, velocity_y_ego, heading_ego,
        #  centerline, target_x,, target_y, wall,
        #  center_x_c2, center_y_c2, velocity_y_c2, heading_c2]
        
        self.observation_space = Box(low=np.array([BUILDING_WIDTH,
                                                   0,
                                                   self.init_ego.min_speed,
                                                   0,
                                                   0,
                                                   BUILDING_WIDTH,
                                                   0,
                                                   0,
                                                   BUILDING_WIDTH,
                                                   0,
                                                   self.init_c2.min_speed,
                                                   0]),
                                     high=np.array([MAP_WIDTH - BUILDING_WIDTH,
                                                    MAP_HEIGHT,
                                                    self.init_ego.max_speed,
                                                    2*np.pi,
                                                    LANE_WIDTH / 2. + SIDEWALK_WIDTH, 
                                                    MAP_WIDTH - BUILDING_WIDTH,
                                                    MAP_HEIGHT,
                                                    MAP_WIDTH/2. - BUILDING_WIDTH,
                                                    MAP_WIDTH - BUILDING_WIDTH,
                                                    MAP_HEIGHT,
                                                    self.init_ego.max_speed,
                                                    2*np.pi]),
                                     shape=(12,),
                                     dtype=np.float32)
        
        self.action_space = Box(low=np.float32(np.array([-0.15,-1])),
                                high=np.float32(np.array([0.15,1])))
        
        self.dt = 0.1
        self.T = 20
        
        self.reset()
        
    def reset(self):
        
        # Randomly choose a target for training
        self.active_goal = random.choice([0,1])
        
        self.world = World(self.dt, width = MAP_WIDTH, height = MAP_HEIGHT, ppm = PPM)
           
        self.ego = self.init_ego.copy()
        self.ego.center = Point(BUILDING_WIDTH + SIDEWALK_WIDTH + 2 + np.random.rand()*(2*LANE_WIDTH + LANE_MARKER_WIDTH - 4), self.np_random.rand()* MAP_HEIGHT/10.)
        self.ego.heading += np.random.randn()*0.1
        self.ego.velocity = Point(0, self.np_random.rand()*5)
        
        self.c2 = self.init_c2.copy()
        self.c2.center = Point(37, random.uniform(10, 40))
        self.c2.heading = np.pi/2
        self.c2.velocity = Point(0, random.uniform(7, 9))
        
        self.targets = []
        self.targets.append(Point(BUILDING_WIDTH + SIDEWALK_WIDTH + LANE_WIDTH/2., MAP_HEIGHT))
        self.targets.append(Point(BUILDING_WIDTH + SIDEWALK_WIDTH + 3*LANE_WIDTH/2. + LANE_MARKER_WIDTH, MAP_HEIGHT))
        self.world.add(Painting(self.targets[self.active_goal], Point(2,2),'yellow'))
        
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, MAP_HEIGHT), 'gray64'))

        # lane markers on the road
        for y in np.arange(LANE_MARKER_HEIGHT/2., MAP_HEIGHT - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))

        # arrows on the road
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2.), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2.), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2.), Point(3*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2.), Point(3*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + LANE_MARKER_WIDTH), Point(2*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + LANE_MARKER_WIDTH), Point(2*LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. - LANE_MARKER_WIDTH/2. - LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + 2*LANE_MARKER_WIDTH), Point(LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))
        self.world.add(Painting(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., MAP_HEIGHT/2. + LANE_MARKER_HEIGHT/2. + 2*LANE_MARKER_WIDTH), Point(LANE_MARKER_WIDTH, LANE_MARKER_WIDTH), 'white'))

        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., MAP_HEIGHT/2.), Point(BUILDING_WIDTH, MAP_HEIGHT)))
        
        self.world.add(self.ego)
        self.world.add(self.c2)
        
        self.ego.dist_to_centerline = distance_to_centerline(self.ego.center.x, MAP_WIDTH/2., LANE_WIDTH, MAP_WIDTH)
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
    
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ego.set_control(action[0],action[1])
        
        # Apply ado vehicle rules
        self.ado_rules()
        
        # Vehicle metrics of interest
        # self.ego.ttc = ttc(self.ego.center.x, self.ego.center.y, self.ego.speed, self.c2.center.x, self.c2.center.y, self.c2.speed,)
        # self.ego.tlc = tlc(self.ego.center.x, self.ego.speed, 0.006, MAP_WIDTH/2., LANE_WIDTH, self.ego.size.x, MAP_WIDTH)
        
        self.world.tick()
        
        return self._get_obs(), self._get_reward(), self.collision_exists or self.target_reached or self.ego.y > MAP_HEIGHT or self.world.t >= self.T, {}
        
    def _get_reward(self):
            return reward(self, LANE_WIDTH)
        
    def _get_obs(self):
        
        self.ego.dist_to_centerline = distance_to_centerline(self.ego.center.x, MAP_WIDTH/2., LANE_WIDTH, MAP_WIDTH)
        self.ego.tlc = tlc(self.ego.center.x, self.ego.speed, 0.006, MAP_WIDTH/2., LANE_WIDTH, self.ego.size.x, MAP_WIDTH)
        self.ego.dist_to_target =  self.targets[self.active_goal].distanceTo(self.ego)
        self.ego.dist_to_wall = distance_to_wall(self.ego.center.x, MAP_WIDTH, BUILDING_WIDTH)

        # [x_ego, y_ego, vx_ego, heading_ego, x_ado, y_ado, vx_ado, heading_ado]
        # [x_ego, y_ego, vy_ego, heading_ego, dist_to_centerline_ego,
        #  dist_to_target_ego, dist_to_wall_ego, x_ado, y_ado, vy_ado, heading_ado]
        return np.float32(np.array([self.ego.center.x,
                                    self.ego.center.y,
                                    self.ego.velocity.y,
                                    self.ego.heading,
                                    self.ego.dist_to_centerline,
                                    self.targets[self.active_goal].x,
                                    self.targets[self.active_goal].y,
                                    self.ego.dist_to_wall,
                                    self.c2.center.x,
                                    self.c2.center.y,
                                    self.c2.velocity.y,
                                    self.c2.heading]))

    def ado_rules(self):
        """
        Rules-based driving policy for ado vehicle.
        """
        # Ado vehicle decelerates to a limit when ego vehicle approaches
        if self.ego.center.y - self.c2.center.y < -10 and self.c2.velocity.y > 2:
            self.c2.velocity.y = self.c2.velocity.y - 0.01
            
        # Ado vehicle accelarates after ego vehicle overtakes
        if self.ego.center.y - self.c2.center.y > 10 and self.c2.velocity.y < 3:
            self.c2.velocity.y = self.c2.velocity.y + 0.01
    
    def render(self, mode='rgb'):
        self.world.render()
        
    def write(self, text): # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH - BUILDING_WIDTH+2), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)