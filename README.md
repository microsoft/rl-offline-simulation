# Project OffSim4RL

This repo intends to track several tools and experiments for the Offline Simulation for Reinforcement Learning paper.

## Getting Started
```console
conda create -n offsim4rl python=3.7
conda activate offsim4rl
pip install -e .
```

## Using offsim4rl

At the time of writing this README, the APIs are not ready yet - the snippets below will work as design guidelines when creating the library:

### Creating simulators based on your offline data

#### Option 1) fit (more stateful)

The same object exposes methods to fit to data as well as environment APIs:

```python
import offsim4rl

# dataset contains observations, actions, rewards, terminals, and infos
dataset = ...
action_space = ...

simulator = offsim4rl.simulators.QueueBased()
simulator.fit(dataset)

# The QueueBased simulator exposes standard OpenAI gym interface
simulator.reset()
simulator.step(action_space.sample())
```

Another example:

```python
import offsim4rl

# dataset contains observations, action_dist, actions, rewards, terminals, and infos
dataset = ...
action_space = ...

simulator = offsim4rl.simulators.PerStateRejectionSampling()

# If the algorithm uses state decoding, it's up to the algo's hyperparameters whether to use the whole dataset for state decoding and rejection sampling, or split the data.
simulator.fit(dataset)

simulator.reset()

# Warning: non-standard "step" method.
# To use full power of Per-State Rejection Sampling, your agent needs to reveal its randomness to the simulator.
# TODO: define distribution type.
simulator.step_dist(...action dist...)
```

#### Option 2) more functional-style

One object learns and serves as a factory for the environment.

```python
import offsim4rl

psrs = offsim4rl.simulators.PerStateRejectionSampling()
simulator = psrs.create_env(dataset)

simulator.reset()
simulator.step_dist(...)
```

### Run benchmark for offline simulators
How to run benchmark evaluating methods for offline evaluation of online reinforcement learning algorithms.

```python
import offsim4rl.benchmark #?
```

### Record dataset for a benchmark
How to create a new benchmark with an environment of your choice.

```python
# TODO: make the example simpler...
from offsim4rl.benchmark import EnvironmentRecorder
from spinup import ppo_pytorch as ppo
import torch
import gym

env_fn = lambda : EnvironmentRecorder(gym.make('CartPole-v1'), 'cartpole_ppo_learner.pkl')

ac_kwargs = dict(hidden_sizes=[32,32], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='cartpole', exp_name='cartpole_ppo_learner')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=100, logger_kwargs=logger_kwargs)
```

## Workflow
- Collect data using OpenAI spinning up
- Train HOMER
- Run offline simulation with learners

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
