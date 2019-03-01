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
        self.q_mass = 0.76 #: [kg] mass
        self.Ixx = 0.00232
        self.arm_length = 0.22 # [m]
        self.arm_width = 0.02 # [m]
        self.height = 0.02 # [m]

        # limits
        self.q_maxAngle = np.pi / 2

        ## Load stuff
        self.l_mass = 0.146
        self.cable_length = 0.82
        self.cable_width = 0.01
        self.l_maxAngle = np.pi / 3

        self.Mass = self.q_mass + self.l_mass
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
            np.finfo(np.float32).max, # xl
            np.finfo(np.float32).max, # zl

            1.0, # Sth
            1.0, # Cth
            1.0, # Sphi
            1.0, # Cphi

            np.finfo(np.float32).max, # xl_dot
            np.finfo(np.float32).max, # zl_dot
            np.finfo(np.float32).max, # th_dot
            np.finfo(np.float32).max, # phi_dot
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minU, self.minU]),
            high = np.array([self.maxU, self.maxU]),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype = np.float32
        )

        self.seed()
        self.viewer = None
        # Need to always call the render() method
        self.state = None # yet :v

    def _apply_action(self, u):
        xl, zl, phi, th, xl_dot, zl_dot, phi_dot, th_dot = self.state

        clipped_u = np.clip(u, self.action_space.low, self.action_space.high)
        
        u1, u2 = clipped_u
        F = u1 + u2
        M = (u1 - u2) * self.arm_length

        sdot = np.array([
            xl_dot,
            zl_dot,
            phi_dot,
            th_dot,
            (-F*np.cos(phi - th) - self.q_mass*self.cable_length*th_dot*2) * np.sin(th) / self.Mass,
            (-F*np.cos(phi - th) - self.q_mass*self.cable_length*th_dot*2) * (-np.cos(th)) / self.Mass - self.gravity,
            M / self.Ixx,
            F*np.sin(phi - th) / (self.q_mass * self.cable_length)
        ])

        self.state = sdot * self.dt + self.state

        return clipped_u

    def reset(self):
        """
        Set a random objective position for the load
        sampling a position for the quadrotor and then
        calculating the load position
        """
        limit = Pendrogone.LIMITS - self.cable_length

        pos_load = 2*limit * np.random.rand(2) - limit
        angle = (np.random.rand(1) * 2 - 1) * (np.array([ self.q_maxAngle, self.l_maxAngle ]) - 0.2)
        # angle = [ 0.0, 0.0 ]

        self.state = np.array([
            pos_load[0],
            pos_load[1],
            angle[0],
            angle[1],
            0, 0, 0, 0
        ])
        self.objective = np.array([0.0, 0.0])

        return self.obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @staticmethod
    def transform(x0, angle, xb):
        T = np.array([ [np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)] ])
        return x0 + T.dot(xb)

    @property
    def pos_quad(self):
        """
        Quadrotor position in the inertial frame
        """
        return self.state[0:2] - self.cable_length * self.p

    @property
    def p(self):
        """
        unit vector from quadrotor to the load
        """
        return np.array([ np.sin(self.state[3]), -np.cos(self.state[3]) ])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 800

        xl,zl,phi,theta = self.state[0:4]
        xq,zq = self.pos_quad

        t1_xy = Pendrogone.transform(self.pos_quad,
                                     self.state[2],
                                     np.array([self.arm_length, 0]))
        t2_xy = Pendrogone.transform(self.pos_quad,
                                     self.state[2],
                                     np.array([-self.arm_length, 0]))
        tl_xy = self.state[0:2]

        to_xy = self.objective

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-Pendrogone.LIMITS[0], Pendrogone.LIMITS[0],
                                   -Pendrogone.LIMITS[1], Pendrogone.LIMITS[1])

            # Render quadrotor frame
            ql,qr,qt,qb = -self.arm_length, self.arm_length, self.arm_width, -self.arm_width
            self.frame_trans = rendering.Transform(rotation=phi, translation=(xq,zq))
            frame = rendering.FilledPolygon([(ql,qb), (ql,qt), (qr,qt), (qr,qb)])
            frame.set_color(0, .8, .8)
            frame.add_attr(self.frame_trans)
            self.viewer.add_geom(frame)

            # Render load cable
            ll,lr,lt,lb = -self.cable_width, self.cable_width, 0, -self.cable_length
            self.cable_trans = rendering.Transform(rotation=theta, translation=(xq,zq))
            cable = rendering.FilledPolygon([(ll,lb), (ll,lt), (lr,lt), (lr,lb)])
            cable.set_color(.1, .1, .1)
            cable.add_attr(self.cable_trans)
            self.viewer.add_geom(cable)

            # Render right propeller
            self.t1_trans = rendering.Transform(translation=t1_xy)
            thruster1 = self.viewer.draw_circle(.04)
            thruster1.set_color(.8, .8, 0)
            thruster1.add_attr(self.t1_trans)
            self.viewer.add_geom(thruster1)

            # Render left propeller 
            self.t2_trans = rendering.Transform(translation=t2_xy)
            thruster2 = self.viewer.draw_circle(.04)
            thruster2.set_color(.8, .8, 0)
            thruster2.add_attr(self.t2_trans)
            self.viewer.add_geom(thruster2)

            # Render load
            self.tl_trans = rendering.Transform(translation=tl_xy)
            load = self.viewer.draw_circle(.08)
            load.set_color(.8, .3, .8)
            load.add_attr(self.tl_trans)
            self.viewer.add_geom(load)

            # Render objective point
            self.to_trans = rendering.Transform(translation=to_xy)
            objective = self.viewer.draw_circle(.02)
            objective.set_color(1., .01, .01)
            objective.add_attr(self.to_trans)
            self.viewer.add_geom(objective)

            
        self.frame_trans.set_translation(xq,zq)
        self.frame_trans.set_rotation(phi)
        self.cable_trans.set_translation(xq,zq)
        self.cable_trans.set_rotation(theta)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        self.tl_trans.set_translation(tl_xy[0], tl_xy[1])

        self.to_trans.set_translation(to_xy[0], to_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
