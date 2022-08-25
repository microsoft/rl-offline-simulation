# setup.py: install script for offsim4rl
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
    packages=setuptools.find_packages(include=['offsim4rl']),
    install_requires=[
        "gym",
        "h5py",
        "joblib",
        "matplotlib",
        "seaborn",
        "tensorboardX",
        "torch>=1.10",
        "tqdm"
    ],
    extras_require={
        'sb3': ["stable-baselines3"]
    }
)
