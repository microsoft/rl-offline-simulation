from curses import KEY_LEFT
import gym
import numpy as np
import pygame


from offsim4rl.agents.agent import Agent
from offsim4rl.data import ProbDistribution

class CartPoleInteractor(Agent):
    def __init__(self, action_space: gym.Space):
        pygame.init()
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("action_space must be gym.spaces.Discrete")
        self.action_space = action_space
        self._prob_dist = np.ones(action_space.n) / action_space.n

    def action_dist_type(self):
        return self.action_dist_type

    def begin_episode(self, observation):
        return self._get_action()

    def commit_action(self, action):
        pass

    def step(self, rew, obs):
        return self._get_action()

    def end_episode(self, reward, truncated=False):
        pass

    def _get_action(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 0
                if event.key == pygame.K_RIGHT:
                    return 1

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='human')
    # agent = CartPoleInteractor(env.action_space)
    while True:
        terminated, truncated = False, False
        action = None
        ret = 0
        steps = 0
        obs, _ = env.reset(seed=0)
        env.render()
        while not truncated:
            print(f'vel: {obs[1]}, angle: {obs[2]}, w: {obs[3]}')
            action = input()
            if action == 'a':
                action = 0
            elif action == 'd':
                action = 1
            # pygame.event.pump()
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_LEFT]:
            #     action = 0
            # if keys[pygame.K_RIGHT]:
            #     action = 1

            # for event in pygame.event.get():
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_LEFT:
            #             action = 0
            #         elif event.key == pygame.K_RIGHT:
            #             action = 1

            if action in (0, 1):
                obs, r, terminated, truncated, _ = env.step(action)
                ret += r
                steps += 1
                env.render()

        print(f'num_steps: {steps}, ret: {ret}')
