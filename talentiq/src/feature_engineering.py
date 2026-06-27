import logging
import pandas as pd

#creating a logger object for this file
logger = logging.getLogger(__name__)
# usually log is used like a print statement , but it doesn't prints , it store's, whenever we get error/crash we can check/rectify acc to the log

#creating a new feature "EmployabiltyScore" ---> Formula = 0.3*cgpa + 0.4*skill_score + 0.3*soft_skill_score
def create_employability_score(df: pd.DataFrame) -> pd.DataFrame:
    df["EmployabilityScore"] = (0.3 * df["cgpa"]+ 0.4 * df["skills_score"]+ 0.3 * df["soft_skills_score"])

    #Stores and Prints the process in terminal with the given message 
    logger.info("EmployabilityScore created.")

    return df

#creating a new feature "PortfolioStrength" ---> Formula =0.5*projects + 0.3*hackathons + 0.2*research_papers
def create_portfolio_strength(df: pd.DataFrame) -> pd.DataFrame:

    df["PortfolioStrength"] = (df["projects"] * 0.5+ df["hackathons"] * 0.3+ df["research_papers"] * 0.2)

    logger.info("PortfolioStrength created.")

    return df

#creating a new feature "TechnicalReadiness" ---> Formula = 0.4*programming_language + 0.3*cerfication + 0.3*skill_score
def create_technical_readiness(df: pd.DataFrame) -> pd.DataFrame:
   
    df["TechnicalReadiness"] = (df["programming_languages"] * 0.4+ df["certifications"] * 0.3+ df["skills_score"] * 0.3)

    logger.info("TechnicalReadiness created.")

    return df

#creating a new feature "ExperienceQuality" ---> Formula =0.6*exp_years + 0.4*internships , then normalizing the score ,converting into 0-1 range
def create_experience_quality(df: pd.DataFrame) -> pd.DataFrame:
    

    raw = (df["experience_years"] * 0.6+ df["internships"] * 0.4)
    #normalizing
    denom = raw.max() - raw.min()
    if denom != 0:
        df["ExperienceQuality"] = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        df["ExperienceQuality"] = 0

    logger.info("ExperienceQuality created.")
    return df


#creating a new feature "LearningIndex" ---> Formula =certifcation+research_papers+hackathons
def create_learning_index(df: pd.DataFrame) -> pd.DataFrame:

    df["LearningIndex"] = (df["certifications"]+ df["research_papers"]+ df["hackathons"])

    logger.info("LearningIndex created.")

    return df

#creating a new feature "ProfileCompleteness" ---> which is how many profile details each student filled and stores the count 
def create_profile_completeness(df: pd.DataFrame) -> pd.DataFrame:
    

    columns = [
        "education_level",
        "university_tier",
        "cgpa",
        "internships",
        "projects",
        "programming_languages",
        "certifications",
        "experience_years",
        "hackathons",
        "research_papers",
        "skills_score",
        "soft_skills_score",
        "company_type",
    ]

    df["ProfileCompleteness"] = (
        df[columns]
        .notna()
        .sum(axis=1)
    )

    logger.info("ProfileCompleteness created.")

    return df

#creating a new feature "SkillExperienceGap" ---> Formula = |skillscore - exp_years*10|
def create_skill_experience_gap(df: pd.DataFrame) -> pd.DataFrame:
    df["SkillExperienceGap"] = abs(df["skills_score"] - df["experience_years"] * 10)

    logger.info("SkillExperienceGap created.")

    return df

#A function to run all the create function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
   
    df = create_employability_score(df)
    df = create_portfolio_strength(df)
    df = create_technical_readiness(df)
    df = create_experience_quality(df)
    df = create_learning_index(df)
    df = create_profile_completeness(df)
    df = create_skill_experience_gap(df)

    logger.info("Feature engineering completed.")

    return df

#main function
if __name__ == "__main__":

    from src.preprocessing import preprocess_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    df = preprocess_data()

    df = engineer_features(df)

    print(df.head())