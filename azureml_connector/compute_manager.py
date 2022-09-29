from azureml.core import compute_target
from azureml.core import compute
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


def create_compute_target(ws, compute_name='gpu-nc6', vm_size='Standard_NC6', min_nodes=0, max_nodes=4):
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print('Found existing compute target, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

    return compute_target
