import os
from azureml.core.conda_dependencies import CondaDependencies

from azureml.core import Environment

def create_env(ws, env_name, requirements_file_path, version=None):
    if version is not None:
        env = Environment.get(workspace=ws, name=env_name, version=version)
        return env

    env = Environment(name=env_name)
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.2-cudnn8-ubuntu20.04'
    conda_dependencies = CondaDependencies()
    conda_dependencies.set_python_version('3.7')
    conda_dependencies.add_channel('pytorch')
    conda_dependencies.add_channel('conda-forge')
    conda_dependencies.add_conda_package('pytorch=1.7.1')
    conda_dependencies.add_conda_package('torchvision=0.8.2')
    conda_dependencies.add_conda_package('torchaudio=0.7.2')
    conda_dependencies.add_conda_package('cudatoolkit=11.3')

    for p in Environment.from_pip_requirements(name="myenv", file_path=requirements_file_path).python.conda_dependencies.pip_packages:
        conda_dependencies.add_pip_package(p)

    env.python.conda_dependencies = conda_dependencies
    env.register(ws)
    return env
