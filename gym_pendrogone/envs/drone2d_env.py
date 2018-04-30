import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Drone2dEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.81 #: [m/s2] acceleration
        self.mass = 0.5 #: [kg] mass
        self.Ixx = 0.00025
        self.arm_length = 0.086 # [m]
        self.height = 0.02 # [m]

        # max and min force for each motor
        self.maxF = 2 * self.mass * self.gravity
        self.minF = 0
        self.maxAngle = 90 * math.pi / 180
        self.dt = 0.01
        
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            self.maxAngle * 2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minF, self.minF]),
            high = np.array([self.maxF, self.maxF]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, z, phi, xdot, zdot, phidot = state
        
        u1, u2 = action
        F = u1 + u2
        M = (u2 - u1) * self.arm_length

        sdot = np.array([
            xdot,
            zdot,
            phidot,
            -F * math.sin(phi) / self.mass,
            F * math.cos(phi) / self.mass - self.gravity,
            M / self.Ixx
        ])

        neu_state = sdot * self.dt + np.array(self.state)
        self.state = neu_state

        done = self.state[2] < -self.maxAngle \
               or self.state[2] > self.maxAngle

        done = bool(done)

        reward_x = (neu_state[0] - state[0]) / self.dt
        reward_z = - neu_state[1] ** 2
        reward = reward_x + reward_z
        
        return self.state, reward, done, {}
        
    def reset(self):
        self.state = np.zeros(6)
        return self.state
    
    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        arm_length = 10
        arm_width = 2
        propeller_rad = 3
        world_width = 2
        scale = 1

        x,z,phi = self.state[0:3].tolist()

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds()



        l,r,t,b = -arm_length, arm_length, arm_width, -arm_width
        frame = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        frame.set_color(0, .8, .8)
        self.frame_trans = rendering.Transform(rotation=phi, translation=(x,z))
        frame.add_attr(self.frame_trans)


        self.viewer.add_geom(frame)

        r_propeller = self.viewer.draw_circle(.1)
        r_propeller.set_color(.8, .8, 0)
        rj = rendering.Transform(translation=(x+cos(phi)*arm_length, z+sin(phi)*amr_length))
        r_propeller.add_attr(rj)
        self.viewer.add_geom(r_propeller)

        l_propeller = self.viewer.draw_circle(.1)
        l_propeller.set_color(.8, .8, 0)
        lj = rendering.Transform(translation=(x-cos(phi)*arm_length, z-sin(phi)*amr_length))
        l_propeller.add_attr(lj)
        self.viewer.add_geom(l_propeller)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
