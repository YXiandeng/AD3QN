from gym.envs.registration import register

register(
    id='UAVEnv-v1',
    entry_point='myGym.researchGym.envs.uav_communication_envs:UAVEnv',
)
