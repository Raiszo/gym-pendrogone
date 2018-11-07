from gym.envs.registration import register

register(
    id='DroneZero-v0',
    entry_point='gym_pendrogone.envs:Drone_zero',
    timestep_limit=200
)

register(
    id='PendrogoneZero-v0',
    entry_point='gym_pendrogone.envs:Pendrogone_zero',
    timestep_limit=200
)

