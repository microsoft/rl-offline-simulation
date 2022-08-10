import itertools
import math
import random
import copy
from torch.nn import functional as F

import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from gym import spaces

class MyGridNavi(gym.Env):
    def __init__(self, num_cells=5, num_steps=15, seed=None, discrete_observation=True):
        super().__init__()

        self.seed(seed)
        self.num_cells = num_cells
        self.num_states = num_cells ** 2

        self._max_episode_steps = num_steps
        self.step_count = 0

        self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2,))
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.nA = self.action_space.n
        self.nS = self.num_states
        self.task_dim = 2
        self._discrete_observation = discrete_observation

        # possible starting states
        self.starting_state = (0.0, 0.0)

        # goals can be anywhere except on possible starting states and immediately around it
        self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
        self.possible_goals.remove((0, 0))
        self.possible_goals.remove((0, 1))
        # self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))

        self.task_dim = 2
        self.num_tasks = self.num_states

        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._goal = self.reset_task()

    def reset(self, seed=None):
        self.step_count = 0
        self._env_state = np.array(self.starting_state)
        return self.external_state

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            self._goal = np.array(task)
        return self._goal

    def get_task(self):
        return self._goal.copy()

    @property
    def external_state(self):
        state = self._env_state.copy()
        if self._discrete_observation:
            state = self.coordinate_to_id(state)
        return state
    
    def state_transition(self, action):
        """
        Moving the agent between states
        """
        if action == 1:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action), str((self.action_space, action))

        done = False or (self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1])

        # perform state transition
        old_state = copy.deepcopy(self._env_state)
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute reward
        if done:
            reward = 0.0
        elif self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
            done = True
        else:
            reward = -0.1

        task = self.get_task()
        task_id = self.coordinate_to_id(task)
        info = {
            'task': task, 'task_id': task_id, 
            'z': self.coordinate_to_id(old_state),
            'z_next': self.coordinate_to_id(self._env_state),
            't': self.step_count-1,
        }
        state = self.external_state
        return state, reward, done, info
    
    def coordinate_to_id(self, xy):
        return int(xy[0] + self.num_cells * xy[1])


class MyGridNaviNoise(MyGridNavi):
    def __init__(self, num_cells=5, num_steps=15, seed=None, augmented_state=False):
        super().__init__(num_cells=num_cells, num_steps=num_steps, seed=seed)
        self._discrete_observation = True
        self._random_bits = None
        self._random_bits_multiplier = int(augmented_state) # log2(number of random bits)
        self._random_bits_rng = np.random.default_rng(seed=seed)
        if self._random_bits == 1:
            self._random_bits = 2
        self.nS = self.num_states * self._random_bits_multiplier
    
    def reset(self, seed=None):
        self._random_bits_rng = np.random.default_rng(seed=seed)
        self._random_bits = self._random_bits_rng.integers(self._random_bits_multiplier)
        return super().reset()
    
    def step(self, action):
        # need to draw random bits before calling superclass step where it calls external state (which uses the bits)
        self._random_bits = self._random_bits_rng.integers(self._random_bits_multiplier)
        return super().step(action)
    
    @property
    def external_state(self):
        state = self._env_state.copy()
        if self._discrete_observation:
            state = self.coordinate_to_id(state) + self.num_states * self._random_bits
        else:
            state = np.concatenate((state, [self._random_bits]))
        return state


class MyGridNaviModuloCounter(MyGridNavi):
    def __init__(self, num_cells=5, num_steps=15, seed=None, counter_limit=1):
        super().__init__(num_cells=num_cells, num_steps=num_steps, seed=seed)
        self._discrete_observation = True
        self._counter = None
        self._counter_limit = int(counter_limit)
        self._counter_rng = np.random.default_rng(seed=seed)
        self.nS = self.num_states * self._counter_limit
    
    def reset(self, seed=None):
        self._counter_rng = np.random.default_rng(seed=seed)
        self._counter = self._counter_rng.integers(self._counter_limit)
        return super().reset()
    
    def step(self, action):
        # need to draw random bits before calling superclass step where it calls external state (which uses the bits)
        self._counter = (self._counter + 1) % self._counter_limit
        return super().step(action)
    
    @property
    def external_state(self):
        state = self._env_state.copy()
        if self._discrete_observation:
            state = self.coordinate_to_id(state) + self.num_states * self._counter
        else:
            state = np.concatenate((state, [self._counter]))
        return state


class MyGridNaviCoords(MyGridNavi):
    def __init__(self, num_cells=5, num_steps=15, seed=None):
        super().__init__(num_cells=num_cells, num_steps=num_steps, seed=seed)
        self._discrete_observation = False
        self._rng = np.random.default_rng(seed=seed)
        self._deltas = np.array([0.0, 0.0])
        self.nS = self.num_states

    def reset(self, seed=None):
        self._rng = np.random.default_rng(seed=seed)
        self._deltas = self._rng.uniform(-0.1, 0.1, 2)
        return super().reset()
    
    def step(self, action):
        self._deltas = self._rng.uniform(-0.1, 0.1, 2)
        return super().step(action)
    
    @property
    def external_state(self):
        out = [
            self._env_state[0] / 5 + 0.1 + self._deltas[0],
            self._env_state[1] / 5 + 0.1 + self._deltas[1],
        ]
        return out

class MyGridNaviHistory(MyGridNavi):
    def __init__(self, num_cells=5, num_steps=15, augmented_state=False, seed=None, action_history=0):
        super().__init__(num_cells=num_cells, num_steps=num_steps, augmented_state=augmented_state, seed=seed)
        self.action_history = action_history
        self.prev_actions = [0] * self.action_history # append last, pop first
        self.lagged_state = self._env_state
        self.nS = self.num_states * (self.nA ** len(self.prev_actions))
    
    def step(self, action):
        self.prev_actions.append(action)
        lagged_action = self.prev_actions.pop(0)
        self.lagged_state_transition(lagged_action)
        _, reward, done, info = super().step(action)
        state = self.external_state
        return state, reward, done, info
    
    def reset(self, seed=None):
        super().reset()
        self.prev_actions = [0] * self.action_history
        self.lagged_state = self._env_state.copy()
        return self.external_state
    
    def lagged_state_transition(self, lagged_action):
        if lagged_action == 1:  # up
            self.lagged_state[1] = min([self.lagged_state[1] + 1, self.num_cells - 1])
        elif lagged_action == 2:  # right
            self.lagged_state[0] = min([self.lagged_state[0] + 1, self.num_cells - 1])
        elif lagged_action == 3:  # down
            self.lagged_state[1] = max([self.lagged_state[1] - 1, 0])
        elif lagged_action == 4:  # left
            self.lagged_state[0] = max([self.lagged_state[0] - 1, 0])

        return self.lagged_state
    
    @property
    def external_state(self):
        state_vec = [*self.lagged_state, *self.prev_actions]
        dims = [self.num_cells, self.num_cells] + [self.nA] * len(self.prev_actions)
        # return state_vec
        s = state_vec[-1]
        for i in range(len(dims)-1, 0, -1):
            s *= dims[i-1]
            s += state_vec[i-1]
        return int(s)
