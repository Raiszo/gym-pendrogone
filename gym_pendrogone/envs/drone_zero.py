import numpy as np
import gym

from . import Drone

class Drone_zero(Drone):
    def __init__(self):
        super().__init__()
        self.reward_shape = Drone_zero.normal_dist(0, np.sqrt(0.1))
        
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
            
        return -20 if dead else +1

    def calc_potential(self):
        dist = np.linalg.norm([ self.state[0] - self.objective[0],
                                self.state[1] - self.objective[1] ])

        return - dist

    @staticmethod
    def normal_dist(mu, sigma_2):
        c = 1/np.sqrt(2*np.pi*sigma_2)

        return lambda x : c * np.exp( - (x-mu)**2 / (2*sigma_2) )
    
    def step(self, action):
        # print(action)
        # print(self.state[0:2])
        old_potential = self.potential

        self._apply_action(action)
        obs = self._get_obs()
        alive = float(self.alive_bonus())

        done = alive < 0
        self.potential = potential = self.calc_potential()

        pot_r = 50 * (potential - old_potential)
        # vel_r = - np.array([5e-3, 5e-3, 5e-2]).dot(obs[4::])
        control_r = - 0.01 * np.ones_like(action).dot(action)
        alive_r = alive
        # print('>',potential)
        closer_r = self.reward_shape(potential)
        # closer_r = +20.0 if potential > -0.3 else 0.0

        reward = np.array([pot_r, control_r, alive_r, closer_r])
        reward = np.sum(reward)

        return obs, reward, done, {}
        
    def _old_step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        dist = np.linalg.norm([ obs[2], obs[3]])
        
        alive = float(self.alive_bonus())
        done = alive < 0

        state_r = -np.array([1e-1, 1e-1, 5e-3, 5e-3, 5e-2]) * np.absolute(obs[2::])
        # state_r = -np.array([1.0, 1.0, 5e-3, 5e-3, 5e-2]) * np.absolute(obs[2::])
        
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
        self.potential = self.calc_potential()

        return self._get_obs()
