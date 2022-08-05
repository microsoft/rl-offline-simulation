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
    description="Benchmark evaluating methods for offline evaluation of online reinforcement learning algorithms.",
    license="MIT",
    keywords="offline reinforcement learning RL benchmark simulation evaluation off-policy rejection sampling",

    include_package_data=True,
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=[
        "azureml-core",
        "azureml-sdk",
        "gym==0.21.0",
        "h5py",
        "joblib",
        "matplotlib",
        "seaborn",
        "stable-baselines3",
        "torch",
        "tqdm"
    ],

    # To install extras type: pip install -e .[gpu]
    extras_require={
        'test': ['pytest']
    },
)
