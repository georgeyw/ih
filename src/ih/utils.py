import json
import os
import tempfile
from typing import Union

import dotenv
import torch
import torch.nn as nn
import wandb
from huggingface_hub import HfApi
from prettytable import PrettyTable
from transformer_lens import HookedTransformer, HookedTransformerConfig


def local_env_setup() -> None:
    rel_path = os.path.join(os.path.dirname(__file__), '../..', '.env')
    dotenv.load_dotenv(rel_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']


def colab_env_setup() -> None:
    # mount Google Drive first
    env_path = '/content/drive/MyDrive/induction_heads/env/.env'
    dotenv.load_dotenv(env_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']    


############################
####### config utils #######
############################

def _get_config_path(config_name: str, type: str) -> str:
    if not config_name.endswith('.json'):
        config_name += '.json'
    return os.path.join(os.path.dirname(__file__), f'{type}_configs', config_name)

def read_config(config_name: str, type: str) -> dict:
    path = _get_config_path(config_name, type)
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config


############################
##### Hugging Face API #####
############################

HF_API = HfApi(
    endpoint="https://huggingface.co",
    token=os.environ['HF_API_KEY'] if 'HF_API_KEY' in os.environ else None,
)
HF_MODEL_NAME = 'model.pth'


def create_hf_repo(name: str, **kwargs) -> str:
    repo_name = os.environ['HF_AUTHOR'] + '/' + name
    return HF_API.create_repo(repo_name, **kwargs)


def load_hf_model(model_name: str, checkpoint_name: str = None) -> nn.Module:
    repo_id = model_name
    if not repo_id.startswith(os.environ['HF_AUTHOR']):
        repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    if model_name.startswith(os.environ['HF_AUTHOR']):
        model_name = model_name.split('/')[1]
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    if checkpoint_name is None:
        filename = HF_MODEL_NAME
    else:
        if not checkpoint_name.endswith('.pth'):
            checkpoint_name += '.pth'
        filename = f'checkpoints/{checkpoint_name}'
    model_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=filename)

    config = HookedTransformerConfig(**load_hf_config(model_name, 'model'))
    state_dict = torch.load(model_path)

    model = HookedTransformer(config)
    model.load_state_dict(state_dict)
    return model


def load_hf_config(model_name: str, config_type: str) -> dict:
    repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    filename = None
    files = HF_API.list_files_info(repo_id=repo_id, repo_type='model')
    for file in files:
        if file.path.startswith(f'configs/{config_type}/'):
            if filename is not None:
                raise ValueError(
                    f"Multiple {config_type} configs found in model repo at {repo_id}.")
            filename = file.path

    config_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config


def upload_hf_model_configs(model_name: str,
                            model_config: Union[str, dict],
                            dgp_config: Union[str, dict],
                            train_config: Union[str, dict],
                            ) -> str:
    repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    def get_name_and_config(config: Union[str, dict], type: str) -> (str, dict):
        if isinstance(config, dict):
            name = 'custom.json'
        if isinstance(config, str):
            name = config
            config = read_config(name, type)
        if not name.endswith('.json'):
            name += '.json'
        return name, config

    model_config_name, model_config = get_name_and_config(model_config, 'model')
    dgp_config_name, dgp_config = get_name_and_config(dgp_config, 'dgp')
    train_config_name, train_config = get_name_and_config(train_config, 'train')

    def upload(name: str, config: dict, type: str):
        if name == 'custom.json':
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = os.path.join(temp_dir, name)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f)
                _upload_hf_file(path_or_fileobj=config_path,
                                path_in_repo=f'configs/{type}/{name}',
                                repo_id=repo_id,
                                repo_type='model')
        else:
            config_path = _get_config_path(name, type)
            _upload_hf_file(path_or_fileobj=config_path,
                            path_in_repo=f'configs/{type}/{name}',
                            repo_id=repo_id,
                            repo_type='model')
    
    upload(model_config_name, model_config, 'model')
    upload(dgp_config_name, dgp_config, 'dgp')
    upload(train_config_name, train_config, 'train')


def upload_hf_model(model: nn.Module,
                    model_name: str,
                    checkpoint_name: str = None) -> str:
    """Creates a temporary file to save the model to, then uploads it to Hugging Face. 
    If a checkpoint name is provided, the model will be saved to a checkpoint file with that name."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # save model
        model_path = os.path.join(temp_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # upload to Hugging Face
        repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
        assert HF_API.token is not None, "Missing HF token"
        assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

        if checkpoint_name is None:
            path_in_repo = HF_MODEL_NAME
        else:
            if not checkpoint_name.endswith('.pth'):
                checkpoint_name += '.pth'
            path_in_repo = f'checkpoints/{checkpoint_name}'

        _upload_hf_file(path_or_fileobj=model_path,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type='model')


def _check_model_exists(model_name: str) -> bool:
    models = HF_API.list_models(author=os.environ['HF_AUTHOR'], search=model_name)
    for model in models:
        if model.id == os.environ['HF_AUTHOR'] + '/' + model_name:
            return True
    return False


def _check_file_exists(repo_id: str, path_in_repo: str) -> bool:
    files = HF_API.list_files_info(repo_id=repo_id, repo_type='model')
    for file in files:
        if file.path == path_in_repo:
            return True
    return False


def _upload_hf_file(path_or_fileobj, path_in_repo, repo_id, repo_type, exists_ok=False) -> None:
    if not exists_ok and _check_file_exists(repo_id, path_in_repo):
        raise ValueError(
            f"File already exists at {repo_id}/{path_in_repo}. Set exists_ok=True to overwrite.")
    HF_API.upload_file(path_or_fileobj=path_or_fileobj,
                       path_in_repo=path_in_repo,
                       repo_id=repo_id,
                       repo_type=repo_type)


############################
####### other stuff ########
############################

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params} ({total_params//1e6}M)")
    return total_params
