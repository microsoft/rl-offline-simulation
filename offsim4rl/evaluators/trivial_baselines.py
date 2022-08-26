""" Trivial baselines to compare to. """
import numpy as np
from offsim4rl.data import OfflineDataset
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling

class FollowObservationOnly(PerStateRejectionSampling):
    """ Follow observation queues, but accept the transition irrespective on the action probabilities. """
    def _reject(self, p_new, p_log, a) -> bool:
        # Ignore action probabilities and just accept the transition.
        return False

class FollowActionOnly(PerStateRejectionSampling):
    """ Reject transition based on the action probabilities, but treat all observations the same. """
    def __init__(self, dataset: OfflineDataset, **kwargs):
        super().__init__(dataset, num_states=1, encoder=_DummyEncoder(), **kwargs)

class ServeRandomTransitions(PerStateRejectionSampling):
    """ Just serve random transitions. """
    def __init__(self, dataset: OfflineDataset, **kwargs):
        super().__init__(dataset, num_states=1, encoder=_DummyEncoder(), **kwargs)
    
    def _reject(self, p_new, p_log, a) -> bool:
        # Ignore action probabilities and just accept the transition.
        return False

class _DummyEncoder:
    def encode(self, observation):
        batch_size = observation.shape[0]
        return [0] * batch_size
