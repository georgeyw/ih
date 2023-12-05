from ih.train import train

model_config_name = 'default-L1.json'
dgp_config_name = 'default.json'
train_config_name = 'test.json'

config = {
    'model_config': model_config_name,
    'dgp_config': dgp_config_name,
    'train_config': train_config_name,
    'run_name': None,
    'save_checkpoints': False,
    'device': 'cpu',
}


if __name__ == '__main__':
    losses, model = train(**config)
