import gym

env = gym.make('CartPole-v0')
env.reset()

for i in range(1000):
    obs, r, done, _ = env.step(env.action_space.sample())
    frame = env.render(mode='rgb_array')
    # print(frame.shape)
    if done:
        break
