import gym
import gym_pendrogone

import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pendrogone-v0')
action = 0.55 * 9.81 / 2

a = (action, action)

obs = env.reset()

for _ in range(200):
    obs, r, done, _ = env.step(a)
    # print(obs)
    env.render(mode = 'rgb_array')
