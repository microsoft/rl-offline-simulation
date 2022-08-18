import numpy as np
import itertools

import gym
from offsim4rl.core import RevealedRandomnessEnv
from offsim4rl.data import OfflineDataset

class QueueEvaluator:
    def __init__(
        self,
        dataset: OfflineDataset,
        num_states=None,
        encoder=None,
    ):

        if not isinstance(dataset.observation_space, gym.spaces.Discrete) and num_states is None and encoder is None:
            raise ValueError('QueueEvaluator only supports discrete observation spaces')
        
        if (num_states is None or encoder is None) and (num_states != encoder):
            raise ValueError('num_states and encoder either both need to be None, or both need to be specified')

        if not isinstance(dataset.action_space, gym.spaces.Discrete):
            # TODO: I believe it should be possible to support continuous action spaces, e.g. by using
            #   torch.distributions inside the PSRS implementation.
            raise ValueError('QueueEvaluator currently only supports discrete action spaces')

        self._dataset = dataset

        zs = encoder.encode(dataset.experience['observations'])
        next_zs = encoder.encode(dataset.experience['next_observations'])
        
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

        kwargs = {
            'buffer': legacy_tuples,
            'nA': dataset.action_space.n,
        }

        # Optional arguments - pass only if not None. Otherwise, use default arg values.
        if num_states is not None:
            kwargs['nS']=num_states

        self._impl = QueueEvaluator_impl(**kwargs)
    
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
        next_obs, r, done, info = self._impl.step(action)
        if next_obs is None:
            return None, None, None, None
        return next_obs, r, done, info

    def step_dist(self, action_dist):
        a = action_dist.sample()
        next_obs, r, done, info = self._impl.step(a)
        if next_obs is None:
            return None, None, None, None, None
        return info['a'], next_obs, r, done, info

class QueueEvaluator_impl(object):
    def __init__(self, buffer, nS=25, nA=5):
        self.raw_buffer = buffer # (s, a, r, s', done, p, info), where p is the logging policy's probability
        self.nS = nS
        self.nA = nA
        self._calculate_latent_state() # self.buffer contains (z, s, a, r, z', s', done, p, info)
        self.reset_sampler()
        self.reset()
    
    def _calculate_latent_state(self):
        self.buffer = [(info['z'], s, a, r, info['next_z'], s_, done, p, info) for s, a, r, s_, done, p, info in self.raw_buffer]
    
    def reset_sampler(self, seed=None):
        self.init_queue = [(elt[0], elt[1]) for elt in self.buffer if elt[-1]['t'] == 0]
        np.random.default_rng(seed=seed).shuffle(self.init_queue)
        
        # Construct queues based on the latent state of from-state
        self.queues = {k:list(v) for k,v in itertools.groupby(sorted(self.buffer, key=lambda elt:(elt[0], elt[2])), key=lambda elt:(elt[0], elt[2]))}
        
        # Shuffle queues
        for z in self.queues.keys():
            np.random.default_rng(seed=seed).shuffle(self.queues[z])
    
    def reset(self, seed=None):
        self.rejection_sampling_rng = np.random.default_rng(seed=seed)
        if len(self.init_queue) == 0:
            self.s = None
            return None
        self.z, self.s = self.init_queue.pop(0)
        return self.s
    
    def step(self, a):
        s = self.s
        z = self.z
        a = int(a)
        if len(self.queues[z,a]) == 0:
            return None, None, None, None
        _z, _s, a, r, z_next, s_next, done, p_log, info = self.queues[z,a].pop(0)
        assert z == _z
        self.s = s_next
        self.z = z_next
        return s_next, r, done, {'z': z, 'next_z': z_next, 'a': a, 'p': p_log}
