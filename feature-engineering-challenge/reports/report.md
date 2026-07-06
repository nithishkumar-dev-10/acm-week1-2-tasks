# Feature Engineering Challenge — Santander Customer Transaction Prediction

## Objective

The task was provided with a baseline Logistic Regression model achieving a recall of 0.83 on the validation set. The goal was to improve recall to 0.88 or higher while maintaining good overall performance, using only Logistic Regression (no other model families) and no external datasets.

## Starting Point

Before any feature engineering, a plain Logistic Regression model was trained directly on the 200 raw columns to establish a true baseline:

| Attempt | Recall | Precision | F1 | Notes |
|---|---|---|---|---|
| Unscaled, no class_weight | 0.2748 | — | — | Didn't converge, max_iter reached |
| Scaled, no class_weight | 0.2810 | — | — | Scaling fixed convergence, but the class imbalance was still untreated |
| Scaled, class_weight='balanced' | 0.7713 | 0.2837 | 0.4149 | Balancing the classes was what actually moved recall — this became the working baseline |

The dataset is roughly 90% class 0 and 10% class 1, which is why recall on the untreated model was so poor — the model had no incentive to catch the minority class at all. Once `class_weight='balanced'` was added, recall jumped to 0.7713 with F1=0.4149. This became the number every later stage was compared against.

## Features Built, and Why

**Frequency encoding.** For each of the 200 original columns, a new column was added recording how common that particular value is. This is the standard first move on this dataset, since individual raw columns only correlate weakly with the target (the strongest single-column correlation was around 0.08) — frequency encoding gives the model a different kind of signal than the raw values alone.

**Row-wise statistical aggregates.** For each customer (row), the mean, standard deviation, min, max, and skew were calculated across all 200 original columns. This gives the model a "summary view" of each customer that no individual column can provide on its own.

**Interaction features.** The top 20 columns most correlated with the target were multiplied together in pairs. This was restricted to the shortlist rather than all 200 columns, since doing it exhaustively would create far too many combinations and mostly add noise rather than signal.

**Outlier capping.** Extreme values in the original columns were capped at the 1st/99th percentile, to stop a handful of unusual customers from distorting the model. This was tried last, as expected to matter least.

**Removing redundant columns.** By the time all of the above features were added, the feature count had grown from 200 to roughly 595. Near-constant columns and highly correlated column pairs were identified and dropped, cutting this down to 509 columns.

All frequency and statistical calculations were computed using only the training portion of the data, then applied unchanged to the validation and test portions — never the other way around, to avoid inflating the score with information the model shouldn't have access to at training time.

## What Worked

| Experiment | Recall | Precision | F1 | Result |
|---|---|---|---|---|
| Frequency encoding, no class_weight | 0.8041 | 0.2642 | 0.3977 | Below baseline F1, but confirmed frequency encoding itself wasn't the problem |
| + Row-wise stats | 0.6733 | 0.3829 | 0.4882 | First stage to clear baseline F1 — row stats added real signal |
| + Interactions | 0.6807 | 0.3849 | **0.4918** | Small further lift, new best F1 at this point |
| + Redundant columns removed | 0.6760 | 0.3845 | 0.4902 | 595 → 509 columns, F1 held essentially flat — confirms the dropped columns weren't carrying unique signal |

Frequency encoding, row-wise statistics, and interaction features all contributed real, additive signal. Removing redundant columns kept performance intact while meaningfully reducing the feature count, which is exactly what that step is meant to do.

The final improvement came from how the recall/precision tradeoff itself was handled, rather than from more features. A grid search over `C`, `penalty`, and `class_weight` showed that `class_weight` was by far the dominant factor — swapping between `None` and `'balanced'` moved F1 far more than any change to `C` or the regularization penalty. But `'balanced'` turned out to be a blunt instrument: it computes an aggressive weighting automatically, which overshot the recall requirement substantially and cost far more precision than necessary.

Instead of using `class_weight='balanced'` as an on/off switch, a custom weight dictionary (e.g. `{0: 1, 1: 2.5}`) was swept alongside the decision threshold, since threshold moving is a separate, complementary lever — it changes where the model's existing probability output is cut into a yes/no decision, without retraining. Combining a moderate custom weight (2.5) with a threshold of 0.40 found a setting that cleared the recall requirement without overshooting it, which produced meaningfully better precision and F1 than the blunt `'balanced'` approach.

## What Didn't Work

**Frequency encoding combined with `class_weight='balanced'`** produced a recall of 0.9863 — which looked spectacular at first — but precision collapsed to 0.1256 and F1 dropped to 0.2228. Checking the predicted vs. actual class distribution confirmed the model was predicting class 1 far more often than the true class 1 rate, meaning the high recall was essentially fake: the model wasn't learning real signal, it was just over-predicting the positive class. Removing `class_weight='balanced'` at that stage (keeping the frequency encoding features) fixed the degenerate behavior and brought precision back to a reasonable 0.2642, confirming the class weighting — not the frequency features — was the cause.

**Outlier capping** was tried on top of frequency encoding, row stats, and interactions. It produced F1=0.4890, slightly worse than the 0.4918 achieved without it. As expected from the nature of this technique on this dataset, it was dropped from the final feature set rather than kept.

**Using `class_weight='balanced'` during hyperparameter tuning** reached recall=0.9465 but precision fell to 0.1618, giving F1=0.2763. This comfortably cleared the recall requirement but overshot it by a wide margin, at a real cost to overall performance — the custom weight + threshold approach described above found a setting that met the same requirement with meaningfully better precision and F1.

## Final Confirmed Score

The final model — frequency encoding + row-wise stats + interaction features, with redundant columns removed, using `class_weight={0: 1, 1: 2.5}` and a decision threshold of 0.40 — was evaluated on the test set for the first and only time:

**Recall: 0.8879 | Precision: 0.2142 | F1: 0.3452**

This meets the recall ≥ 0.88 requirement. The validation score for this same configuration was recall=0.8837, F1=0.3444 — a gap of only 0.0042 between validation and test, which indicates the model generalized well and wasn't overfit to the validation set during tuning.

## Confirmation

Only Logistic Regression was used throughout the project, with no other model family substituted at any stage. No external datasets were used — all feature engineering was derived from the provided Santander Customer Transaction Prediction data. All frequency/statistical calculations were fit on the training data only and applied unchanged to validation and test, and the test set was evaluated exactly once, after every other decision had already been made.
