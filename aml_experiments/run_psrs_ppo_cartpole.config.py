import os
from azureml.core import (
    Dataset,
    Datastore,
    Experiment,
    ScriptRunConfig,
    Workspace,
)

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.data import OutputFileDatasetConfig

from azureml_connector import compute_manager, environment_manager

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

ws = Workspace.from_config(
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'aml_ws_configs', 'config.json'),
    auth=InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
)

datastore = ws.get_default_datastore()
# datastore = Datastore.get(ws, 'offsim4rl')
input_ref = Dataset.File.from_files(datastore.path("offsim4rl/cartpole-v1/**")).as_mount()
output_ref = OutputFileDatasetConfig(
    destination=(datastore, "offsim4rl/cartpole-v1")
)

installation_cmds = "pip install -e git+https://github.com/sebastko/spinningup-simple-install.git#egg=spinup && pip install -e . && "

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "python", "./examples/cartpole/psrs_from_expert_heuristic.py",
        "--input_dir", input_ref,
        "--output_dir", output_ref,
        "--env", "CartPole-v1",
        "--cpu", "0",
        "--steps", "1000",
        "--num_iter", "20000",
        "--exp_name", "CartPole-PSRS-ppo",
        "--prefix", "cartpole",
    ],
    compute_target=compute_manager.create_compute_target(ws, "ds3v2"),
    environment=environment_manager.create_env(
        ws,
        "offsim4rl-env",
        os.path.join(root_dir, "requirements.txt"),
    ),
)

exp = Experiment(workspace=ws, name="psrs-cartpole")
exp.submit(config=script_run_config)
