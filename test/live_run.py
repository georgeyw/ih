import os

from ih.train import train
from ih.utils import HF_API
from ih.utils import local_env_setup

TEST_REPO = 'live-run-test'

model_config_name = 'default-L1.json'
dgp_config_name = 'default.json'
train_config_name = 'test.json'

config = {
    'model_config_name': model_config_name,
    'dgp_config_name': dgp_config_name,
    'train_config_name': train_config_name,
    'run_name': TEST_REPO,
    'save_checkpoints': True,
    'device': 'cpu',
}

# delete HF test repo if it exists
local_env_setup()
repo_name = f'{os.environ["HF_AUTHOR"]}/{TEST_REPO}'
HF_API.delete_repo(repo_name, missing_ok=True)

losses, model = train(**config)
