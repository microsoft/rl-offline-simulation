def test_spinup_import():
    from offsim4rl.agents.spinup.algos.ppo import ppo
    assert callable(ppo)
