import os
import shutil
import gym

from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger

from offsim4rl.agents.ppo import PPOAgent

TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '.test_output')

def test_ppo_agent():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    seed = 0
    env = gym.make('CartPole-v1')

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

    obs, _ = env.reset(seed=seed)
    a, _ = agent.begin_episode(obs)
    for t in range(5000):
        obs, r, terminated, truncated, _ = env.step(a)
        a, _ = agent.step(r, obs, terminated, truncated)

        if terminated or truncated:
            agent.end_episode(r)
            obs, _ = env.reset(seed=seed)
            a, _ = agent.begin_episode(obs)

    log_output_dir = os.path.join(TEST_OUTPUT_DIR, exp_name, f'{exp_name}_s{seed}')
    assert os.path.exists(log_output_dir)
    assert os.path.exists(os.path.join(log_output_dir, 'pyt_save', 'model0.pt'))
    assert os.path.exists(os.path.join(log_output_dir, 'config.json'))
    assert os.path.exists(os.path.join(log_output_dir, 'progress.txt'))
    assert os.path.exists(os.path.join(log_output_dir, 'vars0.pkl'))

    with open(os.path.join(log_output_dir, 'progress.txt'), 'r') as f:
        lines = f.readlines()

    avg_ep_rets = [line.split('\t')[1] for line in lines[1:]]
    expected_avg_ep_rets = ['20.978724', '28.411764', '35.142857', '43.875', '67.14286']
    assert avg_ep_rets == expected_avg_ep_rets

    shutil.rmtree(TEST_OUTPUT_DIR)
    assert not os.path.exists(TEST_OUTPUT_DIR)
