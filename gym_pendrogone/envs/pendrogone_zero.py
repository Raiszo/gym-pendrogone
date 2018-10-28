import math
import numpy as np
import gym

from . import Pendrogone

LIMITS = np.array([1.5, 1.5])
T = 0.02

class Pendrogone_zero(Pendrogone):
    # def __init__(self):
    #     super()__init__()

    def _get_load_pos(self):
        load_pos = Pendrogone.transform( self.state[0:2],
                                         self.state[3],
                                         np.array([0, -self.cable_length]) )

        return load_pos
    
    def _get_obs(self, load_pos):
        x, z, phi, th, xdot, zdot, phidot, thdot = self.state # th := theta

        load_target_theta = np.arctan2( load_pos[0] - self.objective[0],
                                        load_pos[1] - self.objective[1] )

        angle_2_target = load_target_theta - th
        
        return np.array([
            np.sin(th), np.cos(th),
            np.sin(phi), np.cos(phi),
            np.sin(angle_2_target), np.cos(angle_2_target),
            xdot, zdot,
            thdot,
            phidot,
        ], dtype=np.float32)

    def step(self, action):
        self._apply_action(action)

        load_pos = self._get_load_pos()
        obs = self._get_obs(load_pos)
        
        done = self.state[2] < -self.q_maxAngle \
               or self.state[2] > self.q_maxAngle \
               or self.state[3] < -self.l_maxAngle \
               or self.state[3] > self.l_maxAngle

        done = bool(done)
        # done = False

        load_target_dist = np.linalg.norm([ load_pos[0] - self.objective[0],
                                            load_pos[1] - self.objective[1] ])
        
        load_target_cost = -2.0 * load_target_dist
        control_cost = -0.01 * np.ones_like(action).dot(action**2)
        dot_cost = - np.array([0.1, 0.1, 1, 1]).dot(self.state[4:]**2)

        rewards = [
            load_target_dist,
            control_cost,
            dot_cost
        ]
        
        return obs, sum(rewards), done, {}

    def _apply_action(self, u):
        x, z, phi, th, xdot, zdot, phidot, thdot = self.state

        u1, u2 = u
        F = u1 + u2
        M = (u2 - u1) * self.arm_length

        sdot = np.array([
            xdot,
            zdot,
            phidot,
            thdot,
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*th*2) * np.sin(th) / self.Mass,
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*th*2) * np.sin(th) / self.Mass - self.gravity,
            M / self.Ixx,
            np.sin(phi - th) / (self.qmass * self.cable_length)
        ])

        neu_state = sdot * self.dt + self.state
        self.state = neu_state

    
    def reset(self):
        """
        Set a random objective position for the load
        sampling a position for the quadrotor and then
        calculating the load position
        """
        limit = LIMITS - self.cable_length
        mean = np.mean(limit)
        
        q_abs = 2*limit * np.random.rand(2) - mean
        phi = np.random.rand(1) * 2*self.q_maxAngle - self.q_maxAngle
        theta = np.random.rand(1) * 2*self.l_maxAngle - self.l_maxAngle
        
        l_rel = np.array([0. -self.cable_length])
        l_abs = Pendrogone.transform(q_abs, theta, l_rel)

        state = np.array([
            q_abs[0],
            q_abs[1],
            phi,
            theta
        ])
        self.state = np.concatenate((state, np.zeros(4)))
        self.objective = np.array([0.0, 0.0])

        
        load_pos = self._get_load_pos()
        return self._get_obs(load_pos)

    
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
    
