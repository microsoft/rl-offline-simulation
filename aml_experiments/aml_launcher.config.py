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

# interactive_auth = InteractiveLoginAuthentication(
#     tenant_id="TENANT_ID"
# )
# ws = Workspace(
#     subscription_id="SUBSCRIPTION_ID",
#     resource_group="RESOURCE_GROUP",
#     workspace_name="WOKSPACE_NAME",
#     auth=interactive_auth,
# )

interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
ws = Workspace(subscription_id="d347a80f-7b72-4114-96cc-b2faa4ed5344",
               resource_group="fevieira-rg",
               workspace_name="fevieira-aml-ws",
               auth=interactive_auth
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
        installation_cmds + "python", "./examples/experience_collection/random_agent_continuous_grid_rollout.py",
        "--output_dir", "./outputs",
    ],
    compute_target=compute_manager.create_compute_target(ws, "ds3v2"),
    environment=environment_manager.create_env(
        ws,
        "offsim4rl-env",
        os.path.join(root_dir, "requirements.txt"),
    ),
)

exp = Experiment(workspace=ws, name="psrs-grid")
exp.submit(config=script_run_config)
