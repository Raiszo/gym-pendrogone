import numpy as np
import gym

from . import Drone

class Drone2dEnv(Drone):
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
