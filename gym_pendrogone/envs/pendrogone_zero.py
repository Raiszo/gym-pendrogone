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

        angle_2_target = load_target_theta - phi

        return np.array([
            z,
            np.sin(angle_2_target), np.cos(angle_2_target),
            xdot, zdot,
            thdot, phidot,
            np.sin(th), np.cos(th),
            np.sin(phi), np.cos(phi),
        ], dtype=np.float32)

    def alive_bonus(self):
        # dead = self.state[2] < -self.q_maxAngle \
        #     or self.state[2] > self.q_maxAngle \
        #     or self.state[3] < -self.l_maxAngle \
        #     or self.state[3] > self.l_maxAngle \
        #     or self.state[1] < -(LIMITS[1]+0.5)
        dead = self.state[1] < -(LIMITS[1]+0.5)

        return +20 if not dead else -100

    def calc_potential(self, load_pos):
        dist = np.linalg.norm([ load_pos[0] - self.objective[0],
                                load_pos[1] - self.objective[1] ])
        return - dist / T

    def step(self, action):
        self._apply_action(action)

        load_pos = self._get_load_pos()
        obs = self._get_obs(load_pos)

        alive = float(self.alive_bonus())
        done = alive < 0
        # done = False
        
        old_potential = self.potential
        self.potential = self.calc_potential(load_pos)
        progress = float(self.potential - old_potential)

        control_cost = -0.01 * np.sum(action**2)
        stability = 1 - (-self.potential/0.5)**0.4
        # dot_cost = - np.array([0.01, 0.01, 0.01, 0.01]).dot(self.state[4:]**2)

        rewards = [
            # progress,
            # control_cost,
            # dot_cost,
            # stability,
            # alive
            -self.state[1]**2
        ]
        # print(obs)
        # print(rewards)
        
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
        # phi = np.random.rand(1) * 2*self.q_maxAngle - self.q_maxAngle
        # theta = np.random.rand(1) * 2*self.l_maxAngle - self.l_maxAngle
        phi = 0.0
        theta = 0.0
        
        l_rel = np.array([0.0, -self.cable_length])
        # print(l_rel)
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
        self.potential = self.calc_potential(load_pos)

        return self._get_obs(load_pos)

    
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
    
