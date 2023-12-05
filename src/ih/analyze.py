import datetime
from typing import Union

import torch
import wandb
from transformer_lens.utils import lm_cross_entropy_loss
from devinterp.slt.llc import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD

from ih.constants import CHECKPOINT_ZERO_PAD
from ih.constants import WANDB_ENTITY
from ih.constants import WANDB_PROJECT
from ih.dgp import build_dgp_for_model
from ih.utils import load_hf_model
from ih.utils import read_config


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

    model_config = read_config(model_config_name, 'model')
    name += f"L{model_config['n_layers']}"
    name += f"W{model_config['d_model']}"

    if incl_date:
        name += '_'
        name += datetime.datetime.now().strftime("%m-%d")

    # check that other configs exist
    read_config(dgp_config_name, 'model')
    read_config(train_config_name, 'model')

    return name

def _custom_collate(batch):
    batch = torch.stack(batch)
    return [batch, batch.clone()]

def _build_llc_dataloader(dataset, training_cfg, num_workers=0):

    return torch.utils.data.DataLoader(dataset,
                                        batch_size=training_cfg['batch_size'],
                                        collate_fn=_custom_collate,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)

        # tokens = [item for item in batch]
        # tokens_tensor = torch.stack(tokens)
        # return [tokens_tensor, tokens_tensor.clone()]

def estimate_llc_from_hf(repo_id: str, 
                         checkpoint_step: int,
                         dgp_config: Union[str, dict],
                         batch_size: int,
                         num_chains: int, 
                         num_draws: int,
                         learning_rate: float,
                         elasticity: float,
                         num_samples: int,
                         num_workers: int = 0,
                         device: str = 'cpu',
                         seed: int = None):
    # load model from HF
    checkpoint_name = f"checkpoint_{checkpoint_step:0>{CHECKPOINT_ZERO_PAD}d}"
    model = load_hf_model(repo_id, checkpoint_name=checkpoint_name)
    model.cfg.seed = seed

    # build DGP
    if isinstance(dgp_config, str):
        dgp_config = read_config(dgp_config, 'dgp')
    dgp = build_dgp_for_model(model, dgp_config, num_draws)

    # build dataloader
    loader = _build_llc_dataloader(dgp, {'batch_size': batch_size}, num_workers=num_workers)

    # build optimizer
    optim_kwargs = dict(
        lr=learning_rate,
        noise_level=1.0,
        elasticity=elasticity,
        num_samples=num_samples,
        temperature="adaptive",
    )

    results = estimate_learning_coeff_with_summary(
        model=model,
        loader=loader,
        criterion=lm_cross_entropy_loss,
        sampling_method=SGLD,
        optimizer_kwargs=optim_kwargs,
        num_chains=num_chains,
        num_draws=num_draws,
        device=device,
        online=True,
    )

    return results

