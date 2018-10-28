from gym.envs.registration import register

register(
    id='Drone2d-v0',
    entry_point='gym_pendrogone.envs:Drone2dEnv',
    timestep_limit=1000
)

register(
    id='PendrogoneZero-v0',
    entry_point='gym_pendrogone.envs:Pendrogone_zero',
)

