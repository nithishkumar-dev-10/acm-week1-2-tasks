
import yaml
import pandas as pd
import os

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_features():
    with open("config/features.yaml", "r") as f:
        return yaml.safe_load(f)

def load_hyperparams():
    with open("config/hyperparameters.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data():
    
    cfg  = load_config()
    path = cfg["paths"]["raw_data"]
    mode = cfg.get("mode", "full")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at: {path}\n"
            f"        Place resume_dataset.csv inside data/raw/"
        )

    if mode == "sample":
        n  = cfg.get("sample_size", 5000)
        df = pd.read_csv(path, nrows=n)
        print(f"[INFO] Mode=SAMPLE → loaded {len(df)} rows")
    else:
        df = pd.read_csv(path)
        print(f"[INFO] Mode=FULL → loaded {len(df)} rows")

    return df

if __name__ == "__main__":
    cfg  = load_config()
    feat = load_features()
    hp   = load_hyperparams()

    print(f"config.yaml      : {list(cfg.keys())}")
    print(f"features.yaml    : {list(feat.keys())}")
    print(f"hyperparams.yaml : {list(hp.keys())}")
    print(f"Current mode     : {cfg.get('mode', 'full').upper()}")

    df = load_data()
    print(f"Dataset shape    : {df.shape}")
