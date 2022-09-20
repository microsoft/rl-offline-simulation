import os
import shutil
import gym

from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger

from offsim4rl.agents.ppo import PPOAgentRevealed, PPOAgent
from offsim4rl.utils.prob_utils import sample_dist

TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '.test_output')

def test_ppo_agent():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    seed = 0
    env = gym.make('CartPole-v1', new_step_api=True)

    exp_name = 'test-ppo'
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=TEST_OUTPUT_DIR)
    logger = EpochLogger(**logger_kwargs)

    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        ac_kwargs=dict(hidden_sizes=[64] * 2),
        logger=logger,
        seed=seed,
        steps_per_epoch=1000,
    )

    obs = env.reset(seed=seed)
    a = agent.begin_episode(obs)
    for t in range(5000):
        obs, r, terminated, truncated, _ = env.step(a)
        a = agent.step(r, obs)

        if terminated or truncated:
            agent.commit_action(a)
            agent.end_episode(r, truncated=truncated)
            obs = env.reset(seed=seed)
            a = agent.begin_episode(obs)

    log_output_dir = os.path.join(TEST_OUTPUT_DIR, exp_name, f'{exp_name}_s{seed}')
    assert os.path.exists(log_output_dir)
    assert os.path.exists(os.path.join(log_output_dir, 'pyt_save', 'model0.pt'))
    assert os.path.exists(os.path.join(log_output_dir, 'config.json'))
    assert os.path.exists(os.path.join(log_output_dir, 'progress.txt'))
    assert os.path.exists(os.path.join(log_output_dir, 'vars0.pkl'))

    with open(os.path.join(log_output_dir, 'progress.txt'), 'r') as f:
        lines = f.readlines()

    avg_ep_rets = [line.split('\t')[2] for line in lines[1:]]
    expected_avg_ep_rets = ['22.418604', '28.61111', '36.703705', '45.363636', '63.375']
    assert avg_ep_rets == expected_avg_ep_rets

    shutil.rmtree(TEST_OUTPUT_DIR)
    assert not os.path.exists(TEST_OUTPUT_DIR)

def test_ppo_agent_revealed():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    seed = 0
    env = gym.make('CartPole-v1', new_step_api=True)

    exp_name = 'test-ppo-revealed'
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=TEST_OUTPUT_DIR)
    logger = EpochLogger(**logger_kwargs)

    agent = PPOAgentRevealed(
        env.observation_space,
        env.action_space,
        ac_kwargs=dict(hidden_sizes=[64] * 2),
        logger=logger,
        seed=seed,
        steps_per_epoch=1000,
    )

    obs = env.reset(seed=seed)
    action_dist = agent.begin_episode(obs)
    for t in range(5000):
        a = sample_dist(action_dist)
        agent.commit_action(a)
        obs, rew, terminated, truncated, _ = env.step(a)
        action_dist = agent.step(rew, obs)

        if terminated or truncated:
            a = sample_dist(action_dist)
            agent.commit_action(a)
            agent.end_episode(rew, truncated=truncated)
            obs = env.reset(seed=seed)
            action_dist = agent.begin_episode(obs)

    log_output_dir = os.path.join(TEST_OUTPUT_DIR, exp_name, f'{exp_name}_s{seed}')
    assert os.path.exists(log_output_dir)
    assert os.path.exists(os.path.join(log_output_dir, 'pyt_save', 'model0.pt'))
    assert os.path.exists(os.path.join(log_output_dir, 'config.json'))
    assert os.path.exists(os.path.join(log_output_dir, 'progress.txt'))
    assert os.path.exists(os.path.join(log_output_dir, 'vars0.pkl'))

    with open(os.path.join(log_output_dir, 'progress.txt'), 'r') as f:
        lines = f.readlines()

    avg_ep_rets = [line.split('\t')[2] for line in lines[1:]]
    expected_avg_ep_rets = ['22.418604', '28.61111', '36.703705', '45.363636', '63.375']
    assert avg_ep_rets == expected_avg_ep_rets

    shutil.rmtree(TEST_OUTPUT_DIR)
    assert not os.path.exists(TEST_OUTPUT_DIR)
