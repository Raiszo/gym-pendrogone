import numpy as np
import gym

from . import Drone

class Drone_zero(Drone):
    def _get_obs(self):
        x, z, phi, xdot, zdot, phidot = self.state

        load_target_theta = np.arctan2( x - self.objective[0],
                                        z - self.objective[1] )

        angle_2_target = phi - load_target_theta

        return np.array([
            angle_2_target,
            np.sin(phi), np.cos(phi),
            xdot, zdot,
            phidot
        ])
        
    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        
        
        
        done = self.state[2] < -self.maxAngle \
               or self.state[2] > self.maxAngle

        done = bool(done)

        reward_x = (neu_state[0] - state[0]) / self.dt
        reward_z = - neu_state[1] ** 2
        reward = reward_x + reward_z
        
        return self.state, reward, done, {}

    def apply_action(self, u):
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

        
    def reset(self):
        limit = Drone.LIMITS - self.arm_length - 0.05

        phi = 0

        pos = limit * (2*np.random.rand(2) - 1)
        state = np.array([
            pos[0],
            pos[1],
            phi
        ])
        self.state = np.concatenate((state, np.zeros(3)))
        self.objective = np.array([0.0, 0.0])

        return self._get_obs()
