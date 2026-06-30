# TalentIQ — Model Comparison

## Results

| Model | Accuracy | F1-macro | Precision | Recall | ROC-AUC | FPR | FNR | Threshold | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8537 | 0.7153 | 0.7262 | 0.7062 | 0.801 | 0.0769 | 0.5106 | 0.72 | Baseline |
| Random Forest | 0.8333 | 0.7116 | 0.6991 | 0.7285 | 0.794 | 0.1174 | 0.4255 | 0.36 | Compare |
| XGBoost | 0.8503 | 0.7165 | 0.7205 | 0.7128 | 0.7963 | 0.085 | 0.4894 | 0.39 | ✅ Winner |

## Misclassification Analysis

This section identifies which class each model struggles with most.
- **FPR** = Not Hired candidates wrongly predicted as Hired (wasted interviews)
- **FNR** = Hired candidates wrongly predicted as Not Hired (missed good talent)

| Model | FPR | FNR | Threshold | Dominant Error |
|---|---|---|---|---|
| Logistic Regression | 0.0769 | 0.5106 | 0.72 | Class 1 (Hired predicted as Not Hired) |
| Random Forest | 0.1174 | 0.4255 | 0.36 | Class 1 (Hired predicted as Not Hired) |
| XGBoost | 0.085 | 0.4894 | 0.39 | Class 1 (Hired predicted as Not Hired) |

**Business Insight:** A high FNR means the model misses good candidates — costly in recruitment. A high FPR wastes interviewer time on unqualified candidates. The selected model **XGBoost** balances both errors best based on F1-macro.

## Model Selection

**Selected model: XGBoost** — highest F1-macro.

F1-macro was chosen as the primary metric because the dataset has class imbalance. Unlike accuracy, F1-macro penalises models that ignore the minority class. ROC-AUC was used as a tiebreaker to measure ranking quality across thresholds. XGBoost outperformed others by achieving the best balance between precision and recall for both Hired and Not Hired classes. Logistic Regression serves as a linear baseline but cannot capture non-linear hiring patterns. Random Forest handles feature interactions well but is sensitive to depth and sampling parameters. XGBoost with gradient boosting typically handles imbalanced structured data best when tuned correctly, making it the expected winner on this dataset.
