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
            np.sin(phi), np.cos(phi),
            x, z,
            # angle_2_target,
            xdot, zdot,
            phidot
        ])

    def alive_bonus(self):
        dead = self.state[2] < -self.maxAngle \
            or self.state[2] > self.maxAngle \
            or np.absolute(self.state[0]) > Drone.LIMITS[0] \
            or np.absolute(self.state[1]) > Drone.LIMITS[1]
            
        return +1 if not dead else -20
        # return 0.0 if not dead else -100
        
    def step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        dist = np.linalg.norm([ obs[2], obs[3]])
        
        alive = float(self.alive_bonus())
        done = alive < 0

        state_r = -np.array([1e-1, 1e-1, 5e-3, 5e-3, 5e-2]) * np.absolute(obs[2::])
        # state_r = -np.array([1.0, 1.0, 5e-3, 5e-3, 5e-2]) * np.absolute(obs[2::])
        control_r = -np.absolute(action)*0.01
        alive_r = np.array([alive])

        reward = np.concatenate((state_r, control_r, alive_r)) \
            if dist > 0.15 else \
               np.concatenate((state_r, control_r))
        
        # reward = np.concatenate((state_r, control_r))
        reward = np.sum(reward)

        return obs, reward , done, {}

    def _apply_action(self, u):
        x, z, phi, xdot, zdot, phidot = self.state

        u1, u2 = u
        F = u1 + u2
        M = (u2 - u1) * self.arm_length

        sdot = np.array([
            xdot,
            zdot,
            phidot,
            -F * np.sin(phi) / self.mass,
            F * np.cos(phi) / self.mass - self.gravity,
            M / self.Ixx
        ])

        neu_state = sdot * self.dt + np.array(self.state)
        self.state = neu_state

        
    def reset(self):
        limit = Drone.LIMITS - self.arm_length - 0.1

        phi = (np.random.rand(1) * 2 - 1) * self.maxAngle/2

        pos = limit * (2*np.random.rand(2) - 1)
        state = np.array([
            pos[0],
            pos[1],
            phi
        ])
        self.state = np.concatenate((state, np.zeros(3)))
        self.objective = np.array([0.0, 0.0])

        return self._get_obs()
