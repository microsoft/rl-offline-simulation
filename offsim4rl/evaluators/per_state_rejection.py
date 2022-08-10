import gym
from offsim4rl.core import RevealedRandomnessEnv
from offsim4rl.data import OfflineDataset
from .psrs import PSRS

class PerStateRejectionSampling(RevealedRandomnessEnv):
    def __init__(
            self,
            dataset: OfflineDataset,
            num_states=None,
            latent_state_func=None):

        if not isinstance(dataset.observation_space, gym.spaces.Discrete) and num_states is None and latent_state_func is None:
            raise ValueError('PerStateRejectionSampling only supports discrete observation spaces')
        
        if (num_states is None or latent_state_func is None) and (num_states != latent_state_func):
            raise ValueError('num_states and latent_state_func either both need to be None, or both need to be specified')

        if not isinstance(dataset.action_space, gym.spaces.Discrete):
            # TODO: I believe it should be possible to support continuous action spaces, e.g. by using
            #   torch.distributions inside the PSRS implementation.
            raise ValueError('PerStateRejectionSampling currently only supports discrete action spaces')

        self._dataset = dataset

        # PSRS expects action probabilities in the one but last element of the tuple.
        legacy_tuples = (
            (
                row.observation,
                row.action,
                row.reward,
                row.next_observation,
                row.terminal,
                row.action_distribution,
                {**row.info, 't': row.step}
            )
            for row in dataset.iterate_row_tuples()
        )

        kwargs = {
            'buffer': legacy_tuples,
            'nA': dataset.action_space.n,
        }

        # Optional arguments - pass only if not None. Otherwise, use default arg values.
        if num_states is not None:
            kwargs['nS']=num_states
        if latent_state_func is not None:
            kwargs['latent_state_func']=latent_state_func

        self._impl = PSRS(**kwargs)
    
    @property
    def observation_space(self):
        return self._dataset.observation_space
    
    @property
    def action_space(self):
        return self._dataset.action_space
    
    def reset(self):
        return self._impl.reset()

    def step(self, action):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support step(). To implement Per-State Rejection ' +
            'Sampling efficiently, your agent needs to reveal its action distribution via the step_dist() method instead.')
    
    def step_dist(self, action_dist):
        next_obs, r, done, info = self._impl.step(action_dist)
        return info['a'], next_obs, r, done, info
