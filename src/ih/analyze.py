import datetime
import wandb

from ih.constants import WANDB_ENTITY, WANDB_PROJECT
from ih.utils import read_dgp_config
from ih.utils import read_model_config
from ih.utils import read_train_config


# TODO(george): finish setting up wandb
def init_wandb(model_config_name: str, dgp_config_name: str, train_config_name: str):
    run_name = _build_run_name(model_config_name, dgp_config_name, train_config_name)
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=run_name)
    print("Initialized wandb")
    print(f"   - using model config: {model_config_name}")
    print(f"   - using dgp config: {dgp_config_name}")
    print(f"   - using train config: {train_config_name}")
    print(f"   - run name: {run_name}")


def _build_run_name(model_config_name: str,
                    dgp_config_name: str,
                    train_config_name: str,
                    short: bool = True,
                    incl_date: bool = False) -> str:
    if model_config_name.endswith('.json'):
        model_config_name = model_config_name[:-5]
    if dgp_config_name.endswith('.json'):
        dgp_config_name = dgp_config_name[:-5]
    if train_config_name.endswith('.json'):
        train_config_name = train_config_name[:-5]
    if short:
        name = f'[{model_config_name},{dgp_config_name},{train_config_name}]'
    else:
        name = f'[model={model_config_name},dgp={dgp_config_name},train={train_config_name}]'

    model_config = read_model_config(model_config_name)
    name += f"L{model_config['n_layers']}"
    name += f"W{model_config['d_model']}"

    if incl_date:
        name += '_'
        name += datetime.datetime.now().strftime("%m-%d")

    # check that other configs exist
    read_dgp_config(dgp_config_name)
    read_train_config(train_config_name)

    return name
