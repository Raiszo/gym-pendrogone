import numpy as np
from . import Pendrogone


class Pendrogone_zero(Pendrogone):
    def __init__(self):
        super().__init__()

    def step(self, action):
        # This first block of code should not change
        old_potential = self.potential
        
        self._apply_action(action)
        
        potential = self.potential
        
        control_r = -0.05 * action.dot(action)
        alive_r = float(self.alive_bonus())
        pot_r = 50 * (potential - old_potential)

        # -potential = distance to the objective
        shape_r = Pendrogone_zero.reward_shaping( -potential,
                                                  np.linalg.norm(self.state[4:6]) )

        reward = np.array([control_r, alive_r, pot_r, shape_r])
        done = alive_r < 0

        return self.obs, reward, done, {}

    def alive_bonus(self):
        dead = np.absolute(self.state[2]) > self.q_maxAngle \
            or np.absolute(self.state[3]) > self.l_maxAngle \
            or np.absolute(self.state[0]) > Pendrogone.LIMITS[0] \
            or np.absolute(self.state[1]) > Pendrogone.LIMITS[1]

        return -150 if dead else +0.5

    @staticmethod
    def reward_shaping(dist, vel):
        # print(dist, vel)
        c = 5
        dist_r = np.exp(- np.abs(3.5*dist)**2)
        vel_r = np.power(np.exp(- np.abs(vel)), np.exp(- 2.5 * np.abs(dist)))
        # vel_r = np.exp(- np.abs(vel))
        # mask = (dist <= 0.2) * (vel > 1)
        result = c * dist_r * vel_r

        # return np.logical_not(mask) * result + mask * -3.0
        return result

    @staticmethod
    def normal_dist(mu, sigma_2):
        c = 1/np.sqrt(2*np.pi*sigma_2)

        return lambda x: c * np.exp(- (x-mu)**2 / (2*sigma_2))

    @staticmethod
    def exponential():
        # return lambda d, v : max(3 - (3*d) ** 0.4, 0.0) * \
        #     (4 - min(4, max(v, 0.001)))/4 ** (1/max(0.1, d))
        # return lambda d : 3*max(1 - (d/4) ** 0.4, 0.0)
        return lambda d : np.exp(- np.abs(d*2))
 
    @property
    def obs(self):
        xl, zl, phi, th, xl_dot, zl_dot, phi_dot, th_dot = self.state

        # quad_angle = np.arctan2(
        #     self.objective[1] + self.cable_length - self.state[1],
        #     self.objective[0] - self.state[0]
        # )

        # angle_2_target = quad_angle - phi

        obs = np.array([
            xl, zl,
            np.sin(th), np.cos(th),
            np.sin(phi), np.cos(phi),
            xl_dot, zl_dot,
            th_dot, phi_dot,
        ])

        return obs

    @property
    def potential(self):
        dist = np.linalg.norm([ self.state[0] - self.objective[0],
                                self.state[1] - self.objective[1]] )

        return - dist

