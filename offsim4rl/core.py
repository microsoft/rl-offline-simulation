import gym
import abc
import numpy as np
from torch.distributions import Distribution
from typing import Tuple

class RevealedRandomnessEnv(gym.Env):
    """Gym environment with an additional step_dist method allowing the agent
    to pass in a distribution over the action space.
    """

    def step_dist(self, action_dist: object) -> Tuple[object, object, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts a distribution over actions and returns a tuple (action, observation, reward, done, info).

        Args:
            action_dist (object): a distribution over actions, provided by the agent.
                Depending on the implementation, this may be a numpy array (for discrete action spaces),
                a torch Distribution object, or a different description of the distribution.

        Returns:
            action (object): an action sampled from the provided distribution
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # NOTE: another option could be to define our own abstraction Distribution that may wrap other types.
        if isinstance(action_dist, Distribution):
            # Return torch action which may be needed by the agent (e.g., to compute the gradient).
            orig_action = action_dist.sample()
            numpy_action = orig_action.cpu().numpy()
        elif isinstance(action_dist, np.ndarray):
            orig_action = np.random.choice(a=np.arange(len(action_dist)), p=action_dist)
            numpy_action = orig_action
        else:
            # NOTE: other types may be handled by overloads of this method.
            raise ValueError("action_dist must be a torch.distributions.Distribution or numpy array")

        # Step result agnostic to the version of the step API (4 tuple elements with 'done' or 5 tuple elements with 'terminated'/'truncated').
        step_result = self.step(numpy_action)
        return orig_action, *step_result


class RevealedRandomnessEnvWrapper(RevealedRandomnessEnv, gym.Wrapper):
    """Wrapper for gym environments that adds an additional step_dist method."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
