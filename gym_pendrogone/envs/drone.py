import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Drone(gym.Env):
    LIMITS = np.array([1.5, 1.5])
    T = 0.02
    
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.81 #: [m/s2] acceleration
        self.mass = 0.5 #: [kg] mass
        self.Ixx = 0.00232
        self.arm_length = 0.1 # [m]
        self.arm_width = 0.02 # [m]
        self.height = 0.02 # [m]

        # max and min force for each motor
        self.maxF = 2 * self.mass * self.gravity
        self.minF = 0
        self.maxAngle = np.pi / 2
        self.dt = Drone.T

        high = np.array([
            1.0,
            1.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            # 1.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minF, self.minF]),
            high = np.array([self.maxF, self.maxF]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            -high,
            high,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    @staticmethod
    def transform(x0, angle, xb):
        T = np.array([ [np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)] ])
        return x0 + T.dot(xb)

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 800

        x, z, phi = self.state[0:3].tolist()

        t1_xy = Drone.transform(self.state[0:2],
                                     self.state[2],
                                     np.array([self.arm_length, 0]))
        t2_xy = Drone.transform(self.state[0:2],
                                     self.state[2],
                                     np.array([-self.arm_length, 0]))

        to_xy = self.objective
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-Drone.LIMITS[0], Drone.LIMITS[0],
                                   -Drone.LIMITS[1], Drone.LIMITS[1])
            
            l,r,t,b = -self.arm_length, self.arm_length, self.arm_width, -self.arm_width
            self.frame_trans = rendering.Transform(rotation=phi, translation=(x,z))
            frame = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            frame.set_color(0, .8, .8)
            frame.add_attr(self.frame_trans)
            self.viewer.add_geom(frame)

            self.t1_trans = rendering.Transform(translation=t1_xy)
            thruster1 = self.viewer.draw_circle(.04)
            thruster1.set_color(.8, .8, 0)
            thruster1.add_attr(self.t1_trans)
            self.viewer.add_geom(thruster1)

            self.t2_trans = rendering.Transform(translation=t2_xy)
            thruster2 = self.viewer.draw_circle(.04)
            thruster2.set_color(.8, .8, 0)
            thruster2.add_attr(self.t2_trans)
            self.viewer.add_geom(thruster2)

            self.to_trans = rendering.Transform(translation=to_xy)
            objective = self.viewer.draw_circle(.02)
            objective.set_color(1., .01, .01)
            objective.add_attr(self.to_trans)
            self.viewer.add_geom(objective)

        self.frame_trans.set_translation(x,z)
        self.frame_trans.set_rotation(phi)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')        

    def close(self):
        if self.viewer: self.viewer.close()
