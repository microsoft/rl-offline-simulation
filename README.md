# Project OffSim4RL

This repo intends to track several tools and experiments for the Offline Simulation for Reinforcement Learning paper.

## Getting Started

Currently, we support Linux Ubuntu (can be via Windows Subsystem for Linux). Other platforms haven't been tested.

1. Make sure the native dependencies are installed:

```console
sudo apt update
sudo apt install libopenmpi-dev
```

2. [Optional] Create a virtual environment, e.g., using conda:

```console
conda create -n offsim4rl python=3.7
conda activate offsim4rl
```

3. Clone this repository and install the library in the development mode:

```console
git clone https://github.com/microsoft/rl-offline-simulation.git
cd rl-offline-simulation
pip install -r requirements.txt
pip install -e .
```

4. [Optional] Run unit tests.

```console
pytest
```

Note: many dependencies are listed in both "requirements.txt" and in "setup.py".
 * The "requirements.txt" file contains exact versions of the dependencies, some of which are required for our test pipeline to pass. If you'd like to make sure everything is running correctly, would like to reproduce our results, or would like to contribute to the project, we recommend installing the dependecies via "pip install -r requirements.txt", before installing the library.
 * The "setup.py" offers more flexibility in terms of versioning dependencies. If you intend to use offsim4rl as a library in your own project, and you'd like to use different versions of some dependencies than the ones we specified in "requirements.txt", you may skip "pip install -r requirements.txt" and run "pip install -e ." directly.


## Contribute

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
