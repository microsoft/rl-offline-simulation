from gym.envs.registration import register

register(
    id='MyGridNaviCoords-v1',
    entry_point='offsim4rl.envs:MyGridNaviCoords',
    kwargs={'num_cells': 5, 'num_steps': 1000},
)
