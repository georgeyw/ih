from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformer_lens.utils import lm_cross_entropy_loss

from ih.utils import create_hf_repo
from ih.utils import upload_hf_model
from ih.utils import upload_hf_model_configs
from ih.utils import read_config
from ih.constants import CHECKPOINT_EVERY
from ih.constants import CHECKPOINT_ZERO_PAD
from ih.constants import LOG_EVERY
from ih.dgp import build_dgp_for_model
from ih.model import build_model


def train(model_config: Union[str, dict],
          dgp_config: Union[str, dict],
          train_config: Union[str, dict],
          run_name: str = None,
          save_checkpoints: bool = False,
          return_checkpoints: bool = False,
          device: str = 'cpu'):
    if save_checkpoints:
        if run_name is None:
            raise ValueError("Must provide run_name if saving checkpoints")
        create_hf_repo(run_name)
        upload_hf_model_configs(model_name=run_name, 
                                model_config=model_config, 
                                dgp_config=dgp_config, 
                                train_config=train_config)
    if isinstance(train_config, str):
        train_config = read_config(train_config, 'train')
    if train_config['seed'] is not None:
        torch.manual_seed(train_config['seed'])
    model = build_model(model_config, device=device)
    num_samples = train_config['batch_size'] * train_config['max_steps']
    dgp = build_dgp_for_model(model, dgp_config, num_samples)
    loader = DataLoader(dgp, batch_size=train_config['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_config['learning_rate'],
                                  weight_decay=train_config['weight_decay'])
    losses, checkpoints = _training_loop(model, 
                                         optimizer, 
                                         loader, 
                                         train_config, 
                                         run_name=run_name,
                                         save_checkpoints=save_checkpoints,
                                         return_checkpoints=return_checkpoints,
                                         device=device)
    
    if save_checkpoints:
        upload_hf_model(model, run_name)
    if return_checkpoints:
        return losses, checkpoints
    return losses, model


def _training_loop(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   data_loader: DataLoader,
                   train_config: dict,
                   run_name: str = None,
                   save_checkpoints: bool = False,
                   return_checkpoints: bool = False,
                   device: str = 'cpu'):
    losses = []
    checkpoints = []
    for epoch in range(train_config['num_epochs']):
        print(f"Starting epoch: {epoch}")
        for c, batch in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            tokens = batch.to(device)
            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if c % LOG_EVERY == 0:
                print(f"Step: {c}, Loss: {loss.item():.4f}")
            if return_checkpoints and c % CHECKPOINT_EVERY == 0:
                checkpoints.append(deepcopy(model))
            if save_checkpoints and c % CHECKPOINT_EVERY == 0:
                upload_hf_model(model, run_name, checkpoint_name=f"checkpoint_{c:0>{CHECKPOINT_ZERO_PAD}d}")
            if c > train_config['max_steps']:
                break
    return losses, checkpoints


def _num_samples(train_config: dict) -> int:
    batch_size = train_config['batch_size']
    epochs = train_config['num_epochs']
    steps = train_config['max_steps']
    return batch_size * epochs * steps
