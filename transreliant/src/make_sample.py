# src/make_sample.py
#
# Generates the small "sample" raw CSV (safe to commit to GitHub) from the
# full raw CSV (kept local only, gitignored). Run this once whenever the
# full dataset changes, so the committed sample stays in sync.
#
# Usage:
#   python make_sample.py

import pandas as pd
from pathlib import Path
from config_loader import load_config


def make_sample(cfg: dict) -> None:
    # NOTE: intentionally reading cfg["data"]["raw"]["full"] / ["sample"]
    # directly here instead of via get_path() — this script's whole job is
    # to bridge full -> sample, so it needs both paths regardless of
    # whatever `mode` happens to be set to.
    full_path = Path(cfg["data"]["raw"]["full"])
    sample_path = Path(cfg["data"]["raw"]["sample"])
    sample_size = cfg.get("sample_size", 500)
    seed = cfg.get("random_seed", 42)

    if not full_path.exists():
        raise FileNotFoundError(
            f"Full raw data not found at {full_path.resolve()}. "
            "This script needs the full dataset available locally to carve "
            "a sample out of it — it isn't meant to run on CI/GitHub."
        )

    df = pd.read_csv(full_path)
    n = min(sample_size, len(df))
    sample_df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)

    print(f"Full dataset:   {len(df)} rows  -> {full_path}")
    print(f"Sample dataset: {n} rows -> {sample_path}")
    print("Sample file is safe to commit. Full file should stay gitignored.")


def main():
    cfg = load_config()
    make_sample(cfg)


if __name__ == "__main__":
    main()
