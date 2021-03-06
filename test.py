import gym
import gym_pendrogone

import numpy as np

action = 0.906 * 9.81 / 2

env = gym.make('PendrogoneZero-v0')

for _ in range(10):
    obs = env.reset()
    for _ in range(200):
        env.render(mode = 'rgb_array')
        a = env.action_space.sample()
        obs, r, done, _ = env.step(np.array([action-1, action+1]))
        if done:
            break
