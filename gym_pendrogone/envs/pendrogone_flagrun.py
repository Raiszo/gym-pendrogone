import numpy as np
from . import Pendrogone


class Pendrogone_flagrun(Pendrogone):
    def __init__(self):
        super().__init__()

    def flag_reposition(self):
        limits = self.limits - self.cable_length
        self.objective = self.random_uniform(low=-limits, high=limits)
            
    def step(self, action):
        # This first block of code should not change
        old_potential = self.potential
        
        self._apply_action(action)
        
        potential = self.potential
        
        control_r = -0.07 * action.dot(action)
        alive_r = float(self.alive_bonus())
        pot_r = 50 * (potential - old_potential)

        # -potential = distance to the objective
        shape_r = Pendrogone_flagrun.reward_shaping( -potential,
                                                     np.linalg.norm(self.state[4:6]) )

        reward = np.array([control_r, alive_r, pot_r, shape_r])
        done = alive_r < 0

        if -self.potential < 0.1:
            self.flag_reposition()

        return self.obs, reward, done, {}

    @staticmethod
    def reward_shaping(dist, vel):
        # print(dist, vel)
        c = 5
        dist_r = np.exp(- 2 * dist)

        return c * dist_r
