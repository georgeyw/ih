import os

from ih.analyze import estimate_llc_from_hf
from ih.utils import local_env_setup

from live_training_run import TEST_REPO


kwargs = {
    'checkpoint_step': 0,
    'dgp_config': 'default.json',
    'batch_size': 8,
    'num_chains': 2,
    'num_draws': 5,
    'learning_rate': 0.001,
    'elasticity': 1.0,
    'num_samples': 1000,
}

if __name__ == '__main__':
    local_env_setup()
    repo_name = f'{os.environ["HF_AUTHOR"]}/{TEST_REPO}'

    results = estimate_llc_from_hf(repo_id=repo_name, **kwargs)
    for key in results:
        print(f"{key}: {results[key]}")
    