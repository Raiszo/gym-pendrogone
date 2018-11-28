import numpy as np
import gym

from . import Pendrogone

class Pendrogone_zero(Pendrogone):
    def __init__(self):
        super().__init__()
        # self.reward_shape = Pendrogone_zero.normal_dist(0, np.sqrt(0.1))
        self.reward_shape = Pendrogone_zero.exponential()

    def _get_load_pos(self):
        load_pos = Pendrogone.transform( self.state[0:2],
                                         self.state[3],
                                         np.array([0, -self.cable_length]) )

        return load_pos
    
    def _get_obs(self, load_pos):
        x, z, phi, th, xdot, zdot, phidot, thdot = self.state # th := theta

        quad_angle = np.arctan2( self.objective[1] + self.cable_length - self.state[1],
                                 self.objective[0] - self.state[0] )

        angle_2_target = quad_angle - phi

        return np.array([
            np.sin(th), np.cos(th),
            np.sin(phi), np.cos(phi),
            x, z,
            # load_pos[0], load_pos[1],
            # np.sin(angle_2_target), np.cos(angle_2_target),
            xdot, zdot,
            thdot, phidot,
        ])

    def alive_bonus(self):
        dead = np.absolute(self.state[2]) > self.q_maxAngle \
            or np.absolute(self.state[3]) > self.l_maxAngle \
            or np.absolute(self.state[0]) > Pendrogone.LIMITS[0] \
            or np.absolute(self.state[1]) > Pendrogone.LIMITS[1]
        
        return -100 if dead else +0.8

    def calc_potential(self, load_pos):
        dist = np.linalg.norm([ load_pos[0] - self.objective[0],
                                load_pos[1] - self.objective[1] ])
        # dist = np.linalg.norm([ self.state[0] - self.objective[0],
        #                         self.state[1] - self.objective[1] ])

        return - dist

    @staticmethod
    def normal_dist(mu, sigma_2):
        c = 1/np.sqrt(2*np.pi*sigma_2)

        return lambda x : c * np.exp( - (x-mu)**2 / (2*sigma_2) )

    @staticmethod
    def exponential():
        # return lambda d, v : max(3 - (3*d) ** 0.4, 0.0) * \
        #     (4 - min(4, max(v, 0.001)))/4 ** (1/max(0.1, d))
        # return lambda d : 3*max(1 - (d/4) ** 0.4, 0.0)
        return lambda d : np.exp(-np.abs(d*2))


    def step(self, action):
        action = np.clip(action, 0, self.maxF)

        old_potential = self.potential

        self._apply_action(action)
        load_pos = self._get_load_pos()
        obs = self._get_obs(load_pos)

        alive = float(self.alive_bonus())
        done = alive < 0
        self.potential = potential = self.calc_potential(load_pos)
        # self.acceleration = 

        pot_dist_r = 100 * (potential - old_potential)
        control_r = -0.05 * np.ones_like(action).dot(action)
        alive_r = alive
        # print(self.state[7])
        # if -potential > 0.1:
        #     closer_r = self.reward_shape(-potential) * \
        #         (1 - min(3, max(np.absolute(self.state[7]), 0.1))/3)**(1/max(-potential, 0.1))
        # else:
        #     closer_r = 2*self.reward_shape(-potential)
        closer_r = self.reward_shape(-potential)

        # closer_r = self.reward_shape(-potential)
        # print(-potential, np.absolute(self.state[2]))

        reward = np.array([pot_dist_r, control_r, alive_r, closer_r])
        # reward = np.sum(reward)
        
        return obs, reward, done, {}

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
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*thdot*2) * np.sin(th) / self.Mass,
            (-F*np.cos(phi - th) - self.qmass*self.cable_length*thdot*2) * (-np.cos(th)) / self.Mass - self.gravity,
            M / self.Ixx,
            F*np.sin(phi - th) / (self.qmass * self.cable_length)
        ])

        neu_state = sdot * self.dt + self.state
        self.state = neu_state

    
    def reset(self):
        """
        Set a random objective position for the load
        sampling a position for the quadrotor and then
        calculating the load position
        """
        limit = Pendrogone.LIMITS - self.cable_length
        mean = np.mean(limit)
        
        q_abs = 2*limit * np.random.rand(2) - mean
        # q_abs = np.array([0.0, 1.0])
        # phi = (np.random.rand(1) * 2 - 1) * self.q_maxAngle - 0.1
        # theta = (np.random.rand(1) * 2 - 1) * self.l_maxAngle - 0.1
        phi = 0.0
        theta = 0.0
        
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
        # self.acceleration = 0.0

        return self._get_obs(load_pos)

    
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
    
