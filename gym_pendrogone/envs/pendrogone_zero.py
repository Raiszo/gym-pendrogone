import numpy as np
from . import Pendrogone


class Pendrogone_zero(Pendrogone):
    def __init__(self):
        super().__init__()
        # self.reward_shape = Pendrogone_zero.normal_dist(0, np.sqrt(0.1))
        # self.reward_shape = Pendrogone_zero.exponential()

    def step(self, action):
        # This first block of code should not change
        old_potential = self.potential
        self._apply_action(action)
        obs = self._get_obs()
        load_pos, load_vel = self._get_load_info()
        self.potential = potential = self.calc_potential()

        self.load_pos = load_pos
        self.load_vel = load_vel

        alive = float(self.alive_bonus())
        done = alive < 0
        control_r = -0.05 * np.ones_like(action).dot(action)
        alive_r = alive

        pot_r = 100 * (potential - old_potential)
        vel = np.linalg.norm([ self.state[4], self.state[5] ])
        # -potential = distance to the objective
        shape_r = self.reward_shaping(-potential, vel)

        reward = np.array([control_r, alive_r, pot_r, shape_r])
        # reward = np.sum(reward)

        return obs, reward, done, {}

    def alive_bonus(self):
        dead = np.absolute(self.state[2]) > self.q_maxAngle \
            or np.absolute(self.state[3]) > self.l_maxAngle \
            or np.absolute(self.state[0]) > Pendrogone.LIMITS[0] \
            or np.absolute(self.state[1]) > Pendrogone.LIMITS[1]

        return -100 if dead else +0.5

    @staticmethod
    def reward_shaping(dist, vel):
        dist_r = np.exp(- np.abs(dist*2))
        vel_r = np.exp(-np.abs(vel)*(1/max(dist, 0.2)))

        return 2 * dist_r * vel_r

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

    def reset(self):
        """
        Set a random objective position for the load
        sampling a position for the quadrotor and then
        calculating the load position
        """
        limit = Pendrogone.LIMITS - self.cable_length

        q_abs = 2*limit * np.random.rand(2) - limit
        # phi = (np.random.rand(1) * 2 - 1) * self.q_maxAngle - 0.1
        # theta = (np.random.rand(1) * 2 - 1) * self.l_maxAngle - 0.1
        phi = 0.0
        theta = 0.0

        self.state = np.array([
            q_abs[0],
            q_abs[1],
            phi,
            theta,
            0, 0, 0, 0
        ])
        self.objective = np.array([0.0, 0.0])
        self.load_pos = Pendrogone.transform(self.state[0:2],
                                             self.state[3],
                                             np.array([0, -self.cable_length]))
        self.load_vel = np.array([0.0, 0.0])

        # Calculate the initial potential
        self.potential = self.calc_potential()

        return self._get_obs()

    def calc_potential(self):
        dist = np.linalg.norm([ self.load_pos[0] - self.objective[0],
                                self.load_pos[1] - self.objective[1]] )

        return - dist

    def _get_load_info(self):
        old_pos = self.load_pos
        load_pos = Pendrogone.transform(self.state[0:2],
                                             self.state[3],
                                             np.array([0, -self.cable_length]))
        load_vel = (load_pos - old_pos) / self.dt

        return load_pos, load_vel

    def _get_obs(self):
        x, z, phi, th, xdot, zdot, phidot, thdot = self.state  # th := theta

        quad_angle = np.arctan2(
            self.objective[1] + self.cable_length - self.state[1],
            self.objective[0] - self.state[0]
        )

        # angle_2_target = quad_angle - phi

        obs = np.array([
            x, z,
            np.sin(th), np.cos(th),
            np.sin(phi), np.cos(phi),
            xdot, zdot,
            thdot, phidot,
            self.load_pos[0], self.load_pos[1]
        ])

        return obs
