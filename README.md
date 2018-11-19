# pendrogone-gym
openAI gym environment for a 2d quadrotor

Created from the example in https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym

# Pre installation
```bash
sudo apt-get install python-opengl
```

# Installation

```bash
pip install -e .
```

If error with pyglet, something like: "invalid literal for int() with base 10: ' '" uninstall all nvidia drivers. This should be an specific error for WSL, since it does not support them.

# Log

#### 18-11-2018
- The reason why PendrogoneZero-v0 seemed impossible was a wrong dynamic equation. Now the trouble is make a clear gradient to the goal
- As suggested by [Ross Story](https://www.youtube.com/watch?v=0R3PnJEisqk), I'm trying a velocity discount in the load angle, to leave less oscillations at the end of the movement.
#### 15-11-2018
- The agent converges to goal!!!!, using an exponential bonus as suggested [here](https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0)
- Enough testing in the drone env, next: add these rewards to the pendrogone env :D

#### 14-11-2018
- First good result, a gaussian bonus with variance 0.1 and a factor of 50 encourages the agente to stabilize in the vecinity of the goal.
- Theres stationary error, the agent does not converge to the goal, possibly because the gradient is zero near the objective.

#### 13-11-2018
- From previous attempts, reward shaping using potential instead of absolute distance to the goal at least encourages the agent to go fordward.
- Additional bonus reward of +20 when potential > -20, only encourages the agent to oscillate around the actual goal, greedy bastard -_-.

#### 07-11-2018
- With just absolute distance, control effort and a +1 alive bonus still drives to agent to kill itself.
- It's like: "Existence is pain, I just want to die". Guess that like in real life, a really distant enormous reward is not enough to motivate the agent to explore, even worse if any action comes with a cost. What one learns about life while doing some simple RL. :)
