import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_path(cfg: dict, *keys: str) -> str:
    """
    Resolve a data path from config, honoring the sample/full `mode` switch.

    Walks cfg through the given keys, e.g. get_path(cfg, "data", "raw").
    If the node found there is a {"sample": ..., "full": ...} dict, returns
    whichever one matches cfg["mode"]. If the node is just a plain string
    (e.g. splits_dir, which isn't mode-dependent), returns it unchanged.

    This is what lets every script stay agnostic about whether we're
    pointing at the small sample CSV (safe to push to GitHub) or the
    full dataset (kept local / gitignored, used for real training).
    """
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


if __name__ == "__main__":
    # quick sanity check when run directly
    cfg = load_config()
    print("Config loaded successfully:")
    print(cfg)
    print(f"\nCurrent mode: {cfg.get('mode')}")
    print("Resolved raw path:", get_path(cfg, "data", "raw"))
    print("Resolved cleaned path:", get_path(cfg, "data", "cleaned"))
    print("Resolved featured path:", get_path(cfg, "data", "featured"))
