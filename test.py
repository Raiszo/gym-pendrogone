import gym
import gym_pendrogone

import numpy as np
import matplotlib.pyplot as plt

# action = 0.55 * 9.81 / 2

env = gym.make('Pendrogone-v0')

for _ in range(10):
    obs = env.reset()
    for _ in range(200):
        env.render(mode = 'rgb_array')
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        if done:
            print('me rompi D;')
            break
        # print(obs)
