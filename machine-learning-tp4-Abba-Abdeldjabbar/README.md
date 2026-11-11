# Work Session 04 – Overfitting, data leakage and model generalization

> My notes as a student discovering these ideas for the first time.

## What I did
Predict `final_cgpa` (the CGPA band converted to its midpoint) from a student survey (1,200 rows).
Everything is in `Tp-04_completed.ipynb`: **A** inspect the data → **B** show 3 kinds of leakage → **C** see overfitting → **D** fix it and compare the final models.
Fixed seed (42), chronological split (first 80% train / last 20% test), all preprocessing inside a `Pipeline`, and the test set used **once**, at the end.

## Main results

| Model | Train RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Baseline (training mean) | 1.269 | 1.236 | -0.003 |
| Linear Regression | 1.090 | 1.058 | 0.266 |
| Lasso (alpha = 0.1) | 1.106 | 1.072 | 0.246 |
| Decision Tree (depth 3) | 1.102 | 1.068 | 0.252 |
| Gradient Boosting (48 trees) | 1.045 | 1.057 | 0.266 |

Compare this with the unpruned decision tree: train RMSE 0, test RMSE 1.95, **worse than the baseline**.

## What I learned

**1. A perfect training score means nothing.**
The first tree had train RMSE = 0 and it looked "perfect", but it was worse than just predicting the mean. It had memorized the students. Now I look at the *gap* between train and test, not at the train score.

**2. How you split the data changes the answer.**
Random split: test RMSE 1.59. Chronological split: 1.95. Same model, same data. The random split lets the model "see the future", so it is too optimistic. The chronological one is the honest one because it is how the model would really be used.

**3. Leakage can be sneaky.**
- *Duplicates:* a student who submits twice can have one copy in train and one in test (50 of 276 test rows). The tree remembered the twin and the score looked better (1.53 vs 1.59). Remove duplicates **before** splitting.
- *Preprocessing:* fitting the scaler on all data changed the KNN score only a little (1.150 vs 1.145). I first thought "so it doesn't matter", but the rule is about principle: in real life you do not have the test data when you scale. The Pipeline does it correctly for free.
- *Target leakage:* `performance_risk_level` is built from the CGPA, so using it would give a nice but fake R².

**4. Overfitting depends on the real flexibility of the model, not on the number of parameters.**
A degree-15 polynomial on `effort_index` hardly overfit (CV RMSE 1.177 → 1.188), but a deep tree went from 1.07 to 1.95. One input + a smooth curve cannot memorize 960 students, a tree with one leaf per student can. This surprised me.

**5. Overfitting starts quite early.**
For the tree, depth 5 was already the turning point (gap 0.28, test worse than the baseline). Depth 3 was best. Simple models were enough.

**6. Removing a feature can help.**
`submission_day` has almost no correlation with the target (0.02), and in a chronological split the test days are outside the training range. Removing it made the validation RMSE lower or equal for the 3 models. `cohort_code` was also not useful (its differences were smaller than the fold-to-fold noise), so I removed it.

**7. Regularization changes coefficients more than scores.**
Degree-5 Linear Regression had coefficients like 66 and -52 that cancel each other. Ridge made them small, and Lasso set **4 of 5 exactly to 0** (it became a straight line). But the RMSE improved only by about 0.008.

**8. Early stopping is simple but needs care.**
Gradient Boosting needed only **48 trees out of 500**. I had to build the validation block myself (the last 15% of the train set) because the built-in option shuffles the data and breaks the time order.

**9. Modest R² is normal here.**
R² = 0.27. The survey is self-reported, answers are in broad ranges, the target has only 4 values, and real causes (exam results, prior grades) are missing. The best single correlation was 0.36. A low score on a clean evaluation is better than a high score on a leaky one.

## Things I noticed
- The test period is "easier" than the train period (baseline RMSE 1.236 vs 1.269), so most final gaps are **negative**. A negative gap does not mean magic, it means the test set has less noise. I only understood this after comparing with the baseline.
- Linear Regression and Gradient Boosting are tied, so I would choose Linear Regression: same score, much simpler.
- In C.1 I printed the test error for each depth, but this was only to *diagnose*. The depth used in D.4 was chosen with `TimeSeriesSplit`.
- Differences of 0.01 to 0.02 in RMSE are inside the CV noise (fold std about 0.09), so I try not to over-interpret them.
- A decision tree can give slightly different numbers between scikit-learn versions (ties in splits), so some tree numbers may differ in the last digits when you re-run.

## What I would try next
- Use `HistGradientBoostingRegressor` and a few more hyperparameters, still with `TimeSeriesSplit`.
- Predict the 4 CGPA bands with classification instead of the midpoint regression.
- Collect better features (previous grades, exam scores): the model is limited by the data, not by the algorithm.
