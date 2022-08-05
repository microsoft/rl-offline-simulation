import gym
from collections import defaultdict
import numpy as np
import offsim4rl.data
from offsim4rl.data import EnvironmentRecorder
from offsim4rl import utils

def test_record():
    env_name = 'MyGridNaviCoords-v1'
    env = gym.make(env_name, seed=0)
    env.reset_task(np.array([4, 4]))
    pi = defaultdict(lambda: np.ones(5) / 5)  # random policy

    env = EnvironmentRecorder(env)

    done = True
    for _ in range(100):
        if done:
            obs = env.reset()
        action_dist = utils.get_uniform_dist(env.action_space)
        obs, reward, done, info = env.step_dist(action_dist)
    
    env.save('mygrid.h5')

def test_psrs():
    dataset = offsim4rl.data.InMemoryDataset()
    simulator = offsim4rl.simulators.psrs.PerStateRejectionSampling(dataset)
    
    simulator.reset()
    simulator.step_dist()

if __name__ == '__main__':
    test_record()