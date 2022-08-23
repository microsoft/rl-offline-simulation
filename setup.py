# setup.py: install script for aps_tools
"""
to install offsim4rl and its dependencies for development work,
run this cmd from the root directory:
    pip install -e .
"""
import setuptools

setuptools.setup(
    name="offsim4rl",
    version="0.1",
    url="https://www.github.com/microsoft/rl-offline-simulation",
    include_package_data=True,
    packages=['offsim4rl', 'azureml_connector'],
    install_requires=[
        "azureml-core",
        "azureml-sdk",
        "gym",
        "h5py",
        "joblib",
        "matplotlib",
        "seaborn",
        "stable-baselines3",
        "tensorboardX",
        "torch>=1.10",
        "tqdm",
        "spinup @ git+https://github.com/shengpu1126/spinningup.git@shengpu1126-patch-1#egg=spinup",
    ],
)
