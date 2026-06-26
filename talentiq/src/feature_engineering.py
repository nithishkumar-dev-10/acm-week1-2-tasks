

import os
import pandas as pd
import numpy as np
import joblib
from src.config_loader import load_config, load_features

#  5 ORIGINAL FEATURES
def add_employability_score(df):
    """0.3*CGPA + 0.4*SkillsScore + 0.3*SoftSkillsScore"""
    df["EmployabilityScore"] = (
        0.3 * df["CGPA"] +
        0.4 * df["SkillsScore"] +
        0.3 * df["SoftSkillsScore"]
    )
    return df

def add_portfolio_strength(df):
    """Projects*0.5 + Hackathons*0.3 + ResearchPapers*0.2"""
    df["PortfolioStrength"] = (
        df["Projects"]       * 0.5 +
        df["Hackathons"]     * 0.3 +
        df["ResearchPapers"] * 0.2
    )
    return df

def add_technical_readiness(df):
    """Languages_count*0.4 + Certifications*0.3 + SkillsScore*0.3"""
    df["TechnicalReadiness"] = (
        df["Languages_count"] * 0.4 +
        df["Certifications"]  * 0.3 +
        df["SkillsScore"]     * 0.3
    )
    return df

def add_experience_quality(df):
    """YearsExperience*0.6 + Internships*0.4 → min-max normalized"""
    raw = df["YearsExperience"] * 0.6 + df["Internships"] * 0.4
    mn, mx = raw.min(), raw.max()
    df["ExperienceQuality"] = (raw - mn) / (mx - mn + 1e-8)
    return df

def add_learning_index(df):
    """Certifications + ResearchPapers + Hackathons"""
    df["LearningIndex"] = (
        df["Certifications"] +
        df["ResearchPapers"] +
        df["Hackathons"]
    )
    return df

# 2 NEW FEATURES 

def add_profile_completeness(df):
    """
    % of non-null AND non-zero fields per row.
    High = candidate filled out profile completely.
    """
    feature_cols = [c for c in df.columns if c != "Hired"]
    df["ProfileCompleteness"] = df[feature_cols].apply(
        lambda row: (row.notna() & (row != 0)).sum() / len(row),
        axis=1
    )
    return df

def add_skill_experience_gap(df):
    """
    abs(SkillsScore - YearsExperience*10)
    Large gap = skills don't match experience level.
    """
    df["SkillExperienceGap"] = abs(df["SkillsScore"] - df["YearsExperience"] * 10)
    return df

# SAVE FEATURE COLUMN LIST 

def save_feature_columns(df, feat_cfg):
    target = feat_cfg["target"]
    feature_cols = [c for c in df.columns if c != target]
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(feature_cols, "artifacts/feature_columns.pkl")
    print(f"[INFO] Saved {len(feature_cols)} feature columns → artifacts/feature_columns.pkl")
    return feature_cols

#  MAIN 
def run_feature_engineering():
    cfg      = load_config()
    feat_cfg = load_features()
    mode     = cfg.get("mode", "full").upper()

    print(f"\n{'='*50}")
    print(f"  FEATURE ENGINEERING  |  Mode: {mode}")
    print(f"{'='*50}")

    cleaned_path = "data/processed/cleaned.csv"
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError("[ERROR] Run preprocessing first. cleaned.csv not found.")

    df = pd.read_csv(cleaned_path)
    print(f"[INFO] Loaded cleaned data → {df.shape}")

    df = add_employability_score(df)
    df = add_portfolio_strength(df)
    df = add_technical_readiness(df)
    df = add_experience_quality(df)
    df = add_learning_index(df)
    df = add_profile_completeness(df)
    df = add_skill_experience_gap(df)

    df.to_csv(cleaned_path, index=False)
    print("[INFO] Updated cleaned.csv with 7 engineered features")

    save_feature_columns(df, feat_cfg)

    new_feats = [
        "EmployabilityScore", "PortfolioStrength", "TechnicalReadiness",
        "ExperienceQuality", "LearningIndex", "ProfileCompleteness", "SkillExperienceGap"
    ]
    print(f"[INFO] Features created: {new_feats}")
    print(f"\n[DONE] Feature engineering complete | Mode: {mode}\n")
    return df

if __name__ == "__main__":
    run_feature_engineering()
