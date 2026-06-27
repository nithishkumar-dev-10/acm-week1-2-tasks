from functools import lru_cache
from pathlib import Path

import yaml

#This intially stores the current file path and the resolve is used find the entire path ,the parent.parent brings you to the root folder(talentiq)
PROJECT_ROOT = Path(__file__).resolve().parent.parent 

#from the project_root dir , we are moving into config 
CONFIG_DIR = PROJECT_ROOT / "config"

# from the root/config dir , we are specifically storing the path of the respective file's
CONFIG_FILE = CONFIG_DIR / "config.yaml"
FEATURES_FILE = CONFIG_DIR / "features.yaml"
HYPERPARAMETERS_FILE = CONFIG_DIR / "hyperparameters.yaml"


#load the file and return it , raise error if the file is missing
def _load_yaml(path: Path) -> dict:
    
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


#loading the config/config.yaml 
@lru_cache(maxsize=None)
def load_config():
    return _load_yaml(CONFIG_FILE)


#loading the config/features.yaml
@lru_cache(maxsize=None)
def load_features():
    return _load_yaml(FEATURES_FILE)


#loading the config/hyperparameters.yaml
@lru_cache(maxsize=None)
def load_hyperparameters():
    return _load_yaml(HYPERPARAMETERS_FILE)

#reloading all the files
def reload_configs():
    load_config.cache_clear()
    load_features.cache_clear()
    load_hyperparameters.cache_clear()