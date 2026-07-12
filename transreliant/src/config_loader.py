import yaml
from pathlib import Path

#loading the config file , if some error in loading raise filenotfound error
def load_config(config_path: str = "config.yaml") -> dict:

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


#checking with mode in the config and returing the file path
def get_path(cfg: dict, *keys: str) -> str:
  
    node = cfg
    for k in keys:
        node = node[k]

    if isinstance(node, dict) and "sample" in node and "full" in node:
        mode = cfg.get("mode", "full")
        if mode not in node:
            raise KeyError(
                f"mode='{mode}' not found in config at {keys} — available options: {list(node.keys())}"
            )
        return node[mode]

    return node

# main function 
if __name__ == "__main__":
    # quick sanity check when run directly
    cfg = load_config()
    print("Config loaded successfully:")
    print(cfg)
    print(f"\nCurrent mode: {cfg.get('mode')}")
    print("Resolved raw path:", get_path(cfg, "data", "raw"))
    print("Resolved cleaned path:", get_path(cfg, "data", "cleaned"))
    print("Resolved featured path:", get_path(cfg, "data", "featured"))
