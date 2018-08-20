import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Pendrogone(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.81 #: [m/s2] acceleration

        ## Quadrotor stuff
        self.qmass = 0.5 #: [kg] mass
        self.Ixx = 0.003
        self.arm_length = 0.086 # [m]
        self.arm_width = 0.02 # [m]
        self.height = 0.02 # [m]
        # limits
        self.q_maxAngle = 90 * math.pi / 180

        ## Load stuff
        self.lmass = 0.05
        self.cable_length = 0.7
        self.cable_width = 0.005
        self.l_maxAngle = 75 * math.pi / 180

        self.Mass = self.qmass + self.lmass
        # max and min force for each motor
        self.maxF = 2 * self.Mass * self.gravity
        self.minF = 0
        self.dt = 0.02

        """
        The state has 8 dimensions:
         xm,zm :quadrotor position
         phi :quadrotor angle
         theta :load angle, for now the tension in the cable 
                is always non zero
        """
        
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            self.q_maxAngle,
            self.l_maxAngle,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
            np.finfo(np.float32).max
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minF, self.minF]),
            high = np.array([self.maxF, self.maxF]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, z, phi, theta, xdot, zdot, phidot, thetadot = state
        
        u1, u2 = action
        F = u1 + u2
        M = (u2 - u1) * self.arm_length

        sdot = np.array([
            xdot,
            zdot,
            phidot,
            thetadot,
            (-F*np.cos(phi - theta) - self.qmass*self.cable_length*theta*2) * np.sin(theta) / self.Mass,
            (-F*np.cos(phi - theta) - self.qmass*self.cable_length*theta*2) * np.sin(theta) / self.Mass - self.gravity,
            M / self.Ixx,
            np.sin(phi - theta) / (self.qmass * self.cable_length)
        ])

        neu_state = sdot * self.dt + np.array(self.state)
        self.state = neu_state

        done = self.state[2] < -self.q_maxAngle \
               or self.state[2] > self.q_maxAngle \
               or self.state[3] < -self.l_maxAngle \
               or self.state[3] > self.l_maxAngle

        done = bool(done)

        reward_x = -(self.state[0] - self.objective[0])**2
        reward_z = -(self.state[1] - self.objective[1])**2
        reward_angles = -10 * self.state[2] - 20*self.state[3]
        reward = reward_x + reward_z + reward_angles
        
        return self.state, reward, done, {}
        
    def reset(self):
        lposs = np.random()
        return self.state
    
    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        screen_width = 500
        screen_height = 500

        x,z,phi = self.state[0:3].tolist()

        T = np.array([[np.cos(phi), -np.sin(phi), x],
                      [np.sin(phi), np.cos(phi), z],
                      [0, 0, 1]])

        bodyFrame = np.array([[self.arm_length, 0 , 1],
                              [-self.arm_length, 0 , 1]])

        worldFrame = T.dot(bodyFrame.T)[:-1,:]
        t1_xy = worldFrame[:,0].tolist()
        t2_xy = worldFrame[:,1].tolist()
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-1,1,-1,1)
            
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

        self.frame_trans.set_translation(x,z)
        self.frame_trans.set_rotation(phi)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')        

    def close(self):
        if self.viewer: self.viewer.close()
