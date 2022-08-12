import gym
import numpy as np
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling
from offsim4rl.utils.prob_utils import get_uniform_dist
from offsim4rl.data import OfflineDataset, ProbDistribution

def test_per_state_rejection_discrete():
    dataset = OfflineDataset(
        observation_space=gym.spaces.Discrete(25),
        action_space=gym.spaces.Discrete(4),
        action_dist_type=ProbDistribution.Discrete,
        observations=np.array([0, 5], dtype=np.int64),
        actions=np.array([0, 1], dtype=np.int64),
        action_distributions=np.full((2, 4), fill_value=0.25, dtype=np.float32),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        next_observations=np.array([5, 7], dtype=np.int64),
        terminals=np.array([False, True], dtype=bool))
    psrs = PerStateRejectionSampling(dataset)
    obs = psrs.reset()
    assert obs.shape == tuple()
    psrs.step_dist(get_uniform_dist(dataset.action_space))