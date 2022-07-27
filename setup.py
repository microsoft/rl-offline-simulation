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
        "azureml-sdk",
        "azureml-core",
        "azureml-sdk",
        "torch",
        "matplotlib",
        "tqdm",
        "stable-baselines3",
        "gym",
        "joblib",
        "seaborn",
        "h5py"
    ],
)
