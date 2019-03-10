import numpy as np
from . import Pendrogone


class Pendrogone_flagrun(Pendrogone):
    def __init__(self):
        super().__init__()

    def flag_reposition(self, first=False):
        self.objective = self.random_uniform(low=-self.LIMITS, high=self.LIMITS)
            
    def step(self, action):
        # This first block of code should not change
        old_potential = self.potential
        
        self._apply_action(action)
        
        potential = self.potential
        
        control_r = -0.07 * action.dot(action)
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

        return -200 if dead else +0.5

    @property
    def potential(self):
        dist = np.linalg.norm([ self.state[0] - self.objective[0],
                                self.state[1] - self.objective[1]] )

        return - dist

