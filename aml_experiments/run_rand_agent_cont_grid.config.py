import os
from azureml.core import (
    Dataset,
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

# datastore = ws.get_default_datastore()
# input_ref = Dataset.File.from_files(datastore.path("offsim4rl/**")).as_mount()
# output_ref = OutputFileDatasetConfig(
#     destination=(datastore, "offsim4rl")
# )

installation_cmds = "pip install -e . && "

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "python", "./examples/experience_collection/random_agent_continuous_grid_rollout.py",
        "--output_dir", "./outputs",
    ],
    compute_target=compute_manager.create_compute_target(ws, "usw2lowpri4t4gpu"),
    environment=environment_manager.create_env(
        ws,
        "offsim4rl-env",
        os.path.join(root_dir, "requirements.txt"),
    ),
)

exp = Experiment(workspace=ws, name="psrs-grid")
exp.submit(config=script_run_config)
