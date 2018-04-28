from gym.envs.registration import register

register(
    id='drone2d',
    entry_point='gym_pendrogone.envs:drone2d',
)

register(
    id='pendrogone',
    entry_point='gym_pendrogone.envs:pendrogone',
)
