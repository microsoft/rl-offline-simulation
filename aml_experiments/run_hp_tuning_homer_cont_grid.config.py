import os
import sys
from azureml.core import (
    Datastore, Dataset, Experiment, ScriptRunConfig, Workspace,
)

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.data import OutputFileDatasetConfig
from azureml.train.hyperdrive import HyperDriveConfig, RandomParameterSampling, BanditPolicy, choice, PrimaryMetricGoal, GridParameterSampling

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__))))
from azureml_connector import compute_manager
from azureml_connector import environment_manager

ws = Workspace.from_config(
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'aml_ws_configs', 'config.json'),
    auth=InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
)

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
datastore = ws.get_default_datastore()
input_ref = Dataset.File.from_files(datastore.path("offsim4rl/**")).as_mount()
output_ref = OutputFileDatasetConfig(
    destination=(datastore, "offsim4rl")
)

installation_cmds = "pip install -e . && "

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "python", "./examples/continuous_grid/train_homer_encoder.py",
        "--input_dir", input_ref,
        "--output_dir", "./logs",
        "--num_epochs", "1000",
        "--batch_size", "64",
        "--weight_decay", "0.0",
        "--lr", "${{search_space.lr}}",
        "--seed", "${{search_space.seed}}",
        "--latent_size", "${{search_space.latent_size}}",
        "--hidden_size", "${{search_space.hidden_size}}",
        "--temperature_decay", "${{search_space.temperature_decay}}",
    ],
    compute_target=compute_manager.create_compute_target(ws, "gpu-nc24"),
    environment=environment_manager.create_env(
        ws,
        "offsim4rl-env",
        os.path.join(root_dir, "requirements.txt"),
    ),
)

param_sampling = GridParameterSampling({
    'lr': choice(1e-3, 3e-4, 5e-5),
    'latent_size': choice(25, 50),
    'hidden_size': choice(32, 64, 128),
    'seed': choice(0, 7, 17, 42, 100),
    'temperature_decay': choice(True, False),
})

hyperdrive_config = HyperDriveConfig(
    run_config=script_run_config,
    hyperparameter_sampling=param_sampling,
    policy=None,
    primary_metric_name='val_loss',
    primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
    max_total_runs=100,
    max_concurrent_runs=4
)

exp = Experiment(workspace=ws, name="psrs-grid")
exp.submit(config=hyperdrive_config)
