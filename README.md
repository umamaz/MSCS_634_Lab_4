# MSCS 634 – Lab 4: Regression Analysis with Regularization

**Name:** Umama Azhar
**Course:** MSCS 634 – Advanced Big Data and Data Mining

## Purpose

This lab compares several regression techniques for predicting disease progression one year after baseline, using the Diabetes dataset from `sklearn.datasets` (442 patients, 10 standardized health measurements). The notebook (`Lab4(1).ipynb`) builds and evaluates:

1. Simple linear regression (BMI only)
2. Multiple regression (all 10 features)
3. Polynomial regression (degrees 1 to 3 on all features, and degrees 1, 3, 10 on BMI for visualization)
4. Ridge and Lasso regression, tested across several alpha values and with cross-validated alpha

Models are evaluated on a held-out 20% test set (89 patients) using MAE, MSE, RMSE and R², with visualizations for each step.

## Results (test set)

| Model | Test R² | Test RMSE |
|---|---|---|
| Simple (BMI only) | 0.2334 | 63.73 |
| Multiple (10 features) | 0.4526 | 53.85 |
| Polynomial degree 2 | 0.4156 | 55.64 |
| Polynomial degree 3 | -5.2120 | 181.42 |
| Ridge (CV alpha) | 0.4543 | 53.77 |
| Lasso (CV alpha) | 0.4714 | 52.92 |

## Key Insights

- **More features helped.** Going from BMI alone to all 10 features raised R² from 0.23 to 0.45. Even so, more than half of the variation in disease progression remains unexplained.
- **Polynomial terms overfit.** Degree 3 on all features reached a train R² of 0.86 but a test R² of -5.21, because 285 features were fit on only 353 training rows. On BMI alone, higher degrees added almost nothing because the relationship is close to linear.
- **Multicollinearity is present.** In plain multiple regression, `s1` (-931) and `s2` (+518) received large opposite-signed coefficients, which indicates these features carry overlapping information.
- **Regularization stabilized the model.** Ridge shrank the correlated coefficients but kept all features. Lasso set `age`, `s2` and `s4` to zero and gave the best test score (R² = 0.4714), so a simpler model did at least as well as the full one.
- **Alpha controls the trade-off.** A tiny alpha behaves like plain regression, a moderate alpha (about 0.1 for Lasso) helped most, and a large alpha (10) caused underfitting (Lasso zeroed all coefficients, R² = -0.012).
- **Strongest predictors:** BMI, `s5` and blood pressure (`bp`).

## Challenges and Decisions

- **Data cleaning:** The dataset has no missing values or duplicates, and its features are already standardized by sklearn, so no cleaning or scaling was needed.
- **Fair comparison:** All models use the same train/test split (`test_size=0.2`, `random_state=42`) so scores can be compared directly.
- **Choosing alpha without peeking at the test set:** Cross-validated Ridge and Lasso (`RidgeCV`, `LassoCV`) pick alpha using only the training data.
- **Lasso convergence:** `max_iter` was increased to 10000 so Lasso converges at small alpha values.
- **Chart scaling:** The degree 3 polynomial score is so negative that it would flatten the charts, so the axes are clipped and the exact values are reported in the tables.
- **Limited test data:** The improvement from Lasso over plain regression is about +0.02 R² on a single split of 89 patients, so it should be read as suggestive, not conclusive.

## Files

- `Lab4(1).ipynb` – full analysis with code, outputs and written observations
- `README.md` – this file
