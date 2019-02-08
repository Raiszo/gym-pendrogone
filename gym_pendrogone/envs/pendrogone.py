import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class Pendrogone(gym.Env):
    LIMITS = np.array([2.5, 2.5])
    T = 0.02

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 1/T
    }

    def __init__(self):
        self.gravity = 9.81 #: [m/s2] acceleration

        ## Quadrotor stuff
        self.qmass = 0.76 #: [kg] mass
        self.Ixx = 0.00232
        self.arm_length = 0.22 # [m]
        self.arm_width = 0.02 # [m]
        self.height = 0.02 # [m]

        # limits
        self.q_maxAngle = np.pi / 2

        ## Load stuff
        self.lmass = 0.146
        self.cable_length = 0.82
        self.cable_width = 0.01
        self.l_maxAngle = np.pi / 3

        self.Mass = self.qmass + self.lmass
        # max and min force for each motor
        self.maxU = self.Mass * self.gravity
        self.minU = 0
        self.dt = Pendrogone.T

        """
        The state has 8 dimensions:
         xm,zm :quadrotor position
         phi :quadrotor angle
         theta :load angle, for now the tension in the cable 
                is always non zero
        """

        high = np.array([
            np.finfo(np.float32).max, # x
            np.finfo(np.float32).max, # z

            1.0, # Sth
            1.0, # Cth
            1.0, # Sphi
            1.0, # Cphi

            np.finfo(np.float32).max, # xdot
            np.finfo(np.float32).max, # zdot
            np.finfo(np.float32).max, # thdot
            np.finfo(np.float32).max, # phidot

            np.finfo(np.float32).max, # load_pos_x
            np.finfo(np.float32).max, # load_pos_z
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minU, self.minU]),
            high = np.array([self.maxU, self.maxU]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None # yet :v
        self.load_pos = None # Better to put it there [ load_x, loady_z ]
        self.load_vel = None # Don't want to use the equations

    def _apply_action(self, u):
        x, z, phi, th, xdot, zdot, phidot, thdot = self.state

        clipped_u = np.clip(u, self.action_space.low, self.action_space.high)
        
        u1, u2 = clipped_u
        F = u1 + u2
        M = (u2 - u1) * self.arm_length

        sdot = np.array([
            xdot,
            zdot,
            phidot,
            thdot,
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*thdot*2) * np.sin(th) / self.Mass,
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*thdot*2) * (-np.cos(th)) / self.Mass - self.gravity,
            M / self.Ixx,
            F*np.sin(phi - th) / (self.qmass * self.cable_length)
        ])

        self.state = sdot * self.dt + self.state

        return clipped_u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @staticmethod
    def transform(x0, angle, xb):
        T = np.array([ [np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)] ])
        return x0 + T.dot(xb)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 800

        x,z,phi,theta = self.state[0:4].tolist()

        t1_xy = Pendrogone.transform(self.state[0:2],
                                     self.state[2],
                                     np.array([self.arm_length, 0]))
        t2_xy = Pendrogone.transform(self.state[0:2],
                                     self.state[2],
                                     np.array([-self.arm_length, 0]))
        tl_xy = Pendrogone.transform(self.state[0:2],
                                     self.state[3],
                                     np.array([0, -self.cable_length]))

        to_xy = self.objective

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-Pendrogone.LIMITS[0], Pendrogone.LIMITS[0],
                                   -Pendrogone.LIMITS[1], Pendrogone.LIMITS[1])
            
            ql,qr,qt,qb = -self.arm_length, self.arm_length, self.arm_width, -self.arm_width
            self.frame_trans = rendering.Transform(rotation=phi, translation=(x,z))
            frame = rendering.FilledPolygon([(ql,qb), (ql,qt), (qr,qt), (qr,qb)])
            frame.set_color(0, .8, .8)
            frame.add_attr(self.frame_trans)
            self.viewer.add_geom(frame)

            ll,lr,lt,lb = -self.cable_width, self.cable_width, 0, -self.cable_length
            self.cable_trans = rendering.Transform(rotation=theta, translation=(x,z))
            cable = rendering.FilledPolygon([(ll,lb), (ll,lt), (lr,lt), (lr,lb)])
            cable.set_color(.1, .1, .1)
            cable.add_attr(self.cable_trans)
            self.viewer.add_geom(cable)
            
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

            self.tl_trans = rendering.Transform(translation=tl_xy)
            load = self.viewer.draw_circle(.08)
            load.set_color(.8, .3, .8)
            load.add_attr(self.tl_trans)
            self.viewer.add_geom(load)

            self.to_trans = rendering.Transform(translation=to_xy)
            objective = self.viewer.draw_circle(.02)
            objective.set_color(1., .01, .01)
            objective.add_attr(self.to_trans)
            self.viewer.add_geom(objective)

            
        self.frame_trans.set_translation(x,z)
        self.frame_trans.set_rotation(phi)
        self.cable_trans.set_translation(x,z)
        self.cable_trans.set_rotation(theta)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        self.tl_trans.set_translation(tl_xy[0], tl_xy[1])

        self.to_trans.set_translation(to_xy[0], to_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
