import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. IncomePerYear
    df["IncomePerYear"] = df["MonthlyIncome"] / (df["YearsAtCompany"] + 1)
    logger.info("IncomePerYear created.")

    # 2. SatisfactionScore — map strings back to numbers then average
    sat_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4,
               "Bad": 1, "Good": 2, "Better": 3, "Best": 4}
    satisfaction_cols = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction",
                                      "RelationshipSatisfaction", "WorkLifeBalance"]
                         if c in df.columns]
    if satisfaction_cols:
        sat_df = df[satisfaction_cols].replace(sat_map).apply(pd.to_numeric, errors="coerce")
        df["SatisfactionScore"] = sat_df.mean(axis=1)
    else:
        df["SatisfactionScore"] = 2.5
    logger.info("SatisfactionScore created.")

    # 3. LoyaltyIndex
    df["LoyaltyIndex"] = df["YearsWithCurrManager"] / (df["TotalWorkingYears"] + 1)
    logger.info("LoyaltyIndex created.")

    # 4. WorkloadScore
    inv_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    if "JobInvolvement" in df.columns:
        job_inv = df["JobInvolvement"].map(inv_map).fillna(2)
    else:
        job_inv = 2
    df["WorkloadScore"] = df["TrainingTimesLastYear"] * job_inv
    logger.info("WorkloadScore created.")

    # 5. CareerGrowthRate
    df["CareerGrowthRate"] = (
        (df["YearsAtCompany"] - df["YearsSinceLastPromotion"]) /
        (df["YearsAtCompany"] + 1)
    )
    logger.info("CareerGrowthRate created.")

    # 6. OverTimeRisk
    if "OverTime" in df.columns:
        df["OverTimeRisk"] = (df["OverTime"] == "Yes").astype(int)
    else:
        df["OverTimeRisk"] = 0
    logger.info("OverTimeRisk created.")

    # 7. TenureStabilityIndex
    df["TenureStabilityIndex"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    logger.info("TenureStabilityIndex created.")

    # 8. OverTimeXSatisfaction — overtime + low satisfaction = high attrition risk
    if "OverTime" in df.columns and "JobSatisfaction" in df.columns:
        overtime_flag = (df["OverTime"] == "Yes").astype(int)
        sat_num = df["JobSatisfaction"].map(sat_map).fillna(2)
        df["OverTimeXSatisfaction"] = overtime_flag * (5 - sat_num)
    else:
        df["OverTimeXSatisfaction"] = 0
    logger.info("OverTimeXSatisfaction created.")

    # 9. StagnationRisk — years since promotion relative to total tenure
    df["StagnationRisk"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    logger.info("StagnationRisk created.")

    # 10. IncomeVsExperience — underpaid relative to experience = attrition signal
    df["IncomeVsExperience"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    logger.info("IncomeVsExperience created.")

    # 11. DistanceXOverTime — long commute + overtime = burnout risk
    if "DistanceFromHome" in df.columns and "OverTime" in df.columns:
        overtime_flag = (df["OverTime"] == "Yes").astype(int)
        df["DistanceXOverTime"] = df["DistanceFromHome"] * overtime_flag
    else:
        df["DistanceXOverTime"] = 0
    logger.info("DistanceXOverTime created.")

    logger.info("Feature engineering completed.")
    return df