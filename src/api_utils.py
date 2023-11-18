import os
import tempfile

import dotenv
import torch
import torch.nn as nn
import wandb
from huggingface_hub import HfApi

from dgp import _get_dgp_config_path
from model import _get_model_config_path
from train import _get_train_config_path


def local_env_setup() -> None:
    rel_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    dotenv.load_dotenv(rel_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']


############################
##### Hugging Face API #####
############################

HF_API = HfApi(
    endpoint="https://huggingface.co",
    token=None,
)


def create_hf_repo(name: str, **kwargs) -> str:
    repo_name = os.environ['HF_AUTHOR'] + '/' + name
    return HF_API.create_repo(repo_name, **kwargs)


def upload_hf_model_configs(model_name: str,
                            model_config_name: str,
                            dgp_config_name: str,
                            train_config_name: str,
                            ) -> str:
    repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    if not model_config_name.endswith('.json'):
        model_config_name += '.json'
    if not dgp_config_name.endswith('.json'):
        dgp_config_name += '.json'
    if not train_config_name.endswith('.json'):
        train_config_name += '.json'

    model_config_path = _get_model_config_path(model_config_name)
    dgp_config_path = _get_dgp_config_path(dgp_config_name)
    train_config_path = _get_train_config_path(train_config_name)

    _upload_hf_file(path_or_fileobj=model_config_path, 
                        path_in_repo=f'configs/model/{model_config_name}', 
                        repo_id=repo_id, 
                        repo_type='model')
    _upload_hf_file(path_or_fileobj=dgp_config_path, 
                        path_in_repo=f'configs/dgp/{dgp_config_name}', 
                        repo_id=repo_id, 
                        repo_type='model')
    _upload_hf_file(path_or_fileobj=train_config_path, 
                        path_in_repo=f'configs/train/{train_config_name}', 
                        repo_id=repo_id, 
                        repo_type='model')


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
            path_in_repo = 'model.pth'
        else:
            if not checkpoint_name.endswith('.pth'):
                checkpoint_name += '.pth'
            path_in_repo = f'checkpoints/{checkpoint_name}.pth'

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
        raise ValueError(f"File already exists at {repo_id}/{path_in_repo}. Set exists_ok=True to overwrite.")
    HF_API.upload_file(path_or_fileobj=path_or_fileobj, 
                       path_in_repo=path_in_repo, 
                       repo_id=repo_id, 
                       repo_type=repo_type)
