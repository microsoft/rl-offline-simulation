import offsim4rl.data

def test_psrs():
    dataset = offsim4rl.data.InMemoryDataset()

    simulator = offsim4rl.simulators.psrs.PerStateRejectionSampling(dataset)
    
    simulator.reset()
    simulator.step_dist()