import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # quick sanity check when run directly
    cfg = load_config()
    print("Config loaded successfully:")
    print(cfg)