# pendrogone-gym
openAI gym environment for a 2d quadrotor

Created from the example in https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym

# Installation

```bash
pip install -e .
```

If error with pyglet, something like: "invalid literal for int() with base 10: ' '" uninstall all nvidia drivers. This should be an specific error for WSL, since it does not support them.