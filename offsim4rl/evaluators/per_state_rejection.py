import time

import gym
from offsim4rl.core import RevealedRandomnessEnv
from offsim4rl.data import OfflineDataset
from .psrs import PSRS
from torch.distributions import Distribution

class PerStateRejectionSampling(RevealedRandomnessEnv):
    def __init__(
        self,
        dataset: OfflineDataset,
        num_states=None,
        encoder=None,
        new_step_api=False
    ):

        if not isinstance(dataset.observation_space, gym.spaces.Discrete) and num_states is None and encoder is None:
            raise ValueError('PerStateRejectionSampling only supports discrete observation spaces')

        if (num_states is None or encoder is None) and (num_states != encoder):
            raise ValueError('num_states and encoder either both need to be None, or both need to be specified')

        if not isinstance(dataset.action_space, gym.spaces.Discrete):
            # TODO: I believe it should be possible to support continuous action spaces, e.g. by using
            #   torch.distributions inside the PSRS implementation.
            raise ValueError('PerStateRejectionSampling currently only supports discrete action spaces')

        self._dataset = dataset
        start_time = time.time()
        if encoder is not None:
            zs = encoder.encode(dataset.experience['observations'])
            next_zs = encoder.encode(dataset.experience['next_observations'])
        else:
            # Assume discrete observations that correspond to states
            zs = dataset.experience['observations']
            next_zs = dataset.experience['next_observations']

        print(f'Time to encode observations: {time.time() - start_time}')
        # PSRS expects action probabilities in the one but last element of the tuple.
        legacy_tuples = (
            (
                row.observation,
                row.action,
                row.reward,
                row.next_observation,
                row.terminal,
                row.action_distribution,
                {**row.info, 't': row.step, 
                 'z': zs[i], 'next_z': next_zs[i]}
            )
            for i, row in enumerate(dataset.iterate_row_tuples())
        )
        print(f'Time to create legacy tuples: {time.time() - start_time}')

        kwargs = {
            'buffer': legacy_tuples,
            'nA': dataset.action_space.n,
            'reject_func': self._reject
        }

        # Optional arguments - pass only if not None. Otherwise, use default arg values.
        if num_states is not None:
            kwargs['nS'] = num_states

        self.new_step_api = new_step_api

        self._impl = PSRS(**kwargs)

    @property
    def observation_space(self):
        return self._dataset.observation_space

    @property
    def action_space(self):
        return self._dataset.action_space

    def reset_sampler(self, seed=None):
        return self._impl.reset_sampler(seed=seed)

    def reset(self, seed=None):
        return self._impl.reset(seed=seed)

    def step(self, action):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support step(). To implement Per-State Rejection ' +
            'Sampling efficiently, your agent needs to reveal its action distribution via the step_dist() method instead.')

    def step_dist(self, action_dist):
        if isinstance(action_dist, Distribution):
            action_dist = action_dist.probs
        next_obs, r, done, info = self._impl.step(action_dist)

        # terminated, truncated
        dones = [done, False] if self.new_step_api else [done]

        if next_obs is None:
            return (None,) * (6 if self.new_step_api else 5)
        return (info['a'], next_obs, r, *dones, info)

    def _reject(self, p_new, p_log, a) -> bool:
        return self._impl._default_reject(p_new, p_log, a)
