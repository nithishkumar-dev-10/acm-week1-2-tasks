# TalentIQ — Model Comparison

## Results

| Model | Accuracy | F1-macro | Precision | Recall | ROC-AUC | FPR | FNR | Verdict |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7755 | 0.6667 | 0.6503 | 0.72 | 0.7909 | 0.1984 | 0.3617 | ✅ Winner |
| Random Forest | 0.8469 | 0.6561 | 0.7077 | 0.6333 | 0.7922 | 0.0526 | 0.6809 | Compare |
| XGBoost | 0.8231 | 0.6589 | 0.6652 | 0.6535 | 0.7873 | 0.0972 | 0.5957 | Compare |

## Misclassification Analysis

This section identifies which class each model struggles with most.
- **FPR** = Not Hired candidates wrongly predicted as Hired (wasted interviews)
- **FNR** = Hired candidates wrongly predicted as Not Hired (missed good talent)

| Model | FPR | FNR | Dominant Error |
|---|---|---|---|
| Logistic Regression | 0.1984 | 0.3617 | Class 1 (Hired predicted as Not Hired) |
| Random Forest | 0.0526 | 0.6809 | Class 1 (Hired predicted as Not Hired) |
| XGBoost | 0.0972 | 0.5957 | Class 1 (Hired predicted as Not Hired) |

**Business Insight:** A high FNR means the model misses good candidates — costly in recruitment. A high FPR wastes interviewer time on unqualified candidates. The selected model **Logistic Regression** balances both errors best based on F1-macro.

## Model Selection

**Selected model: Logistic Regression** — highest F1-macro.

F1-macro was chosen as the primary metric because the dataset has class imbalance. Unlike accuracy, F1-macro penalises models that ignore the minority class. ROC-AUC was used as a tiebreaker to measure ranking quality across thresholds. Logistic Regression outperformed others by achieving the best balance between precision and recall for both Hired and Not Hired classes. Logistic Regression serves as a linear baseline but cannot capture non-linear hiring patterns. Random Forest handles feature interactions well but is sensitive to depth and sampling parameters. XGBoost with gradient boosting typically handles imbalanced structured data best when tuned correctly, making it the expected winner on this dataset.
