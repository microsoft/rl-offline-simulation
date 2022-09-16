def test_spinup_spinup_import():
    from spinup.spinup.algos.pytorch.ppo.ppo import ppo
    assert callable(ppo)

def test_spinup_import():
    from spinup.algos.pytorch.ppo.ppo import ppo
    assert callable(ppo)
