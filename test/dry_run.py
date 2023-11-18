from ih.train import train

model_config_name = 'default-L1.json'
dgp_config_name = 'default.json'
train_config_name = 'test.json'

config = {
    'model_config_name': model_config_name,
    'dgp_config_name': dgp_config_name,
    'train_config_name': train_config_name,
    'run_name': None,
    'save_checkpoints': False,
    'device': 'cpu',
}

losses, model = train(**config)
