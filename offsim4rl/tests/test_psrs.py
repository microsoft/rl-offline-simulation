import gym
import numpy as np
from tqdm import tqdm

from offsim4rl.agents import tabular
from offsim4rl.envs import gridworld
from offsim4rl.evaluators import psrs

def test_psrs_tabular():
    num_logging_episodes = 10
    num_cells = 5
    num_steps = 10

    # The process is random, but usually we can simulate around 80 steps.
    # So if we cannot simulate 40 repeatedly, it's likely that something has been broken.
    min_expected_simulated_steps = 40

    env = gridworld.MyGridNavi(num_cells=num_cells, num_steps=num_steps, seed=0)
    env.reset_task(np.array([4,4]))

    metadata = tabular.rollout(env, num_logging_episodes, tabular.epsilon_greedy_policy(np.zeros((env.nS, env.nA)), dict(epsilon=1)), gamma=1, seed=0)

    env_psrs = psrs.PSRS(metadata['memory'], nS=env.nS, nA=5)
    
    obs = env_psrs.reset()
    steps_simulated = -1
    while obs is not None:
        obs, r, done, info = env_psrs.step(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        steps_simulated += 1
        if done:
            obs = env_psrs.reset()
    
    print(f'Simulated {steps_simulated} steps')
    assert steps_simulated > min_expected_simulated_steps
