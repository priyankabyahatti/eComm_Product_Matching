""" The script holds the initial configuration required when running this project locally."""

# TODO Deployment of the project to be done in future
import os
from pathlib import Path

# make it available globally
print(os.getcwd())
project_dir = os.getcwd()

def create_required_dirs() -> None:
    dirs = ['ml_models', 'results', 'training_data']
    for dir in dirs:
        Path(f"{project_dir}/{dir}").mkdir(parents=True, exist_ok=True)


