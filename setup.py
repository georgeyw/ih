from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ih',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'ih': ['train_configs/*.json',
               'model_configs/*.json',
               'dgp_configs/*.json'],
    },
    install_requires=requirements
)
