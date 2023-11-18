import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_lens.utils import lm_cross_entropy_loss

from constants import LOG_EVERY
from dgp import build_dgp_for_model
from model import build_model


# TODO(george): add wandb logging
def train(model_config_name: str,
          dgp_config_name: str,
          train_config_name: str,
          device: str = 'cuda'):
    train_config = _read_train_config(train_config_name)
    model = build_model(model_config_name)
    dgp = build_dgp_for_model(model, dgp_config_name)
    dataset = dgp.generate_dataset(_num_samples(train_config))
    loader = DataLoader(dataset, batch_size=train_config['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_config['lr'],
                                  weight_decay=train_config['weight_decay'])
    losses = _training_loop(model, optimizer, loader, train_config, device=device)
    return losses


def _training_loop(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   data_loader: DataLoader,
                   train_config: dict,
                   device: str = 'cuda'):
    losses = []
    for epoch in range(train_config['epochs']):
        print(f"Starting epoch: {epoch}")
        for c, batch in enumerate(data_loader):
            optimizer.zero_grad()
            tokens = batch['tokens'].device(device)
            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if c % LOG_EVERY == 0:
                print(f"Step: {c}, Loss: {loss.item():.4f}")
            if c > train_config['max_steps']:
                break
    return losses


def _read_train_config(config_name) -> dict:
    path = _get_train_config_path(config_name)
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config


def _get_train_config_path(config_name: str) -> str:
    if not config_name.endswith('.json'):
        config_name += '.json'
    return os.path.join(os.path.dirname(__file__), 'train_configs', config_name)


def _num_samples(train_config: dict) -> int:
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    steps = train_config['max_steps']
    return batch_size * epochs * steps
