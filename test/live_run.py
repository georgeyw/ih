import os

from ih.train import train
from ih.utils import HF_API
from ih.utils import local_env_setup
from ih.utils import read_config

TEST_REPO = 'live-run-test'

model_config_name = 'default-L1.json'
dgp_config_name = 'default.json'
train_config_name = 'test.json'

config = {
    'model_config': model_config_name,
    'dgp_config': dgp_config_name,
    'train_config': train_config_name,
    'run_name': TEST_REPO,
    'save_checkpoints': True,
    'device': 'cpu',
}

# delete HF test repo if it exists
local_env_setup()
repo_name = f'{os.environ["HF_AUTHOR"]}/{TEST_REPO}'
HF_API.delete_repo(repo_name, missing_ok=True)

losses, model = train(**config)


# rerun with custom configs

TEST_REPO_CUSTOM = 'live-run-custom-test'

model_config = read_config(model_config_name, 'model')
dgp_config = read_config(dgp_config_name, 'dgp')
train_config = read_config(train_config_name, 'train')

config = {
    'model_config': model_config,
    'dgp_config': dgp_config,
    'train_config': train_config,
    'run_name': TEST_REPO_CUSTOM,
    'save_checkpoints': True,
    'device': 'cpu',
}

repo_name = f'{os.environ["HF_AUTHOR"]}/{TEST_REPO_CUSTOM}'
HF_API.delete_repo(repo_name, missing_ok=True)

losses, model = train(**config)