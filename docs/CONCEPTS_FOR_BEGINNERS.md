# Concepts for beginners: predicting loan default with gradient boosting

## 1. The problem
Home Credit lends to people with little or no credit history. For each applicant we know ~120 facts (income, loan size, age, job, documents provided, external credit scores...) and, for past applicants, whether they later **defaulted** (had payment difficulties). The task: given a new applicant, estimate the **probability of default** so that the lender can approve, price or refuse. This is **binary classification** with target `TARGET` = 1 (default) or 0 (repaid).

## 2. Class imbalance (why accuracy is a trap)
Only about **8%** of applicants default. A "model" that predicts *everyone repays* is 92% accurate and completely useless. So we use metrics that handle imbalance: **AUC**, precision, recall, F1, precision-recall curves, lift.

## 3. AUC in plain words
Take one random defaulter and one random non-defaulter. **AUC is the probability that the model gives the defaulter the higher risk score.** 0.5 = coin flip, 1.0 = perfect. It measures *ranking*, not the probability itself, and does not depend on a threshold. (Gini = 2 x AUC - 1; KS is the maximum gap between the two groups' score distributions.)
* **ROC curve**: true positive rate vs false positive rate as the threshold moves. AUC is the area under it.
* **Precision** = of those we flag, how many really default. **Recall** = of all defaulters, how many we flag. **F1** = their harmonic mean.
* **Precision-recall curve** is more informative than ROC when positives are rare.
* **Threshold**: the score above which we say "risky". It is a *business* choice (cost of a missed defaulter vs cost of refusing a good customer), not a modelling detail.

## 4. Decision trees and gradient boosting
A **decision tree** asks a sequence of yes/no questions ("is EXT_SOURCE_3 < 0.4?") and ends in a leaf with a risk estimate. One tree is weak and unstable.
**Gradient boosting** builds hundreds of small trees *one after another*; each new tree is fitted to the **mistakes** (gradients of the loss) of the ensemble so far, and its output is added with a small weight (the **learning rate**). It handles missing values, interactions and non-linearities automatically, and is the state of the art for tabular data.
* **LightGBM, XGBoost, CatBoost** are three fast implementations. CatBoost has special handling for categorical variables; LightGBM is usually fastest.
* **Random forest** instead averages many *independent* deep trees (bagging): more robust, usually less accurate than boosting.
* **Logistic regression** is a linear model: transparent, a good baseline.

## 5. Overfitting and regularisation
A model can memorise the training data. Remedies: limit tree depth/leaves, require a minimum number of samples per leaf (`min_child_samples`), subsample rows and columns, L1/L2 penalties on leaf values, a small learning rate, and **early stopping** (stop adding trees when a validation set stops improving).

## 6. Train / validation / test and cross-validation
* **Hold-out (test) set**: 20% of rows put aside and used **once** at the very end to estimate real-world performance.
* **K-fold cross-validation** on the remaining 80%: split into 5 parts, train on 4 and score on the 5th, rotate. Gives 5 AUCs: the **mean** is the estimate, the **standard deviation** tells us how stable the model is. **Stratified** keeps the 8% default rate in each fold.
* Early stopping needs its own validation slice. We carve it out of the *training* part of each fold, never from the scoring fold, otherwise the CV score would be optimistic.

## 7. Data leakage (the silent killer)
Leakage = using information at training time that would not be available when predicting, or that leaks the answer. Examples: a column computed from the outcome; scaling or imputing using the whole dataset including validation rows; target encoding without nested folds; duplicated people on both sides of a split; an ID number that encodes time. Our checks:
1. **Single-feature AUC scan**: a single column with AUC > 0.8 would be suspicious.
2. **Shuffled-target test**: train on randomly permuted labels; a leak-free pipeline scores ~0.50.
3. **ID check**: AUC of `SK_ID_CURR` alone should be ~0.5.
4. **Adversarial validation**: train a classifier to tell training rows from test rows; high AUC means the two sets differ (distribution shift).
5. Only unsupervised, history-before-application aggregations from the secondary tables.

## 8. Missing values and categories
Many columns are missing (e.g. housing details, external scores). Trees can treat "missing" as its own branch; linear models need imputation (median) and categories need one-hot encoding. Missingness itself can be informative. The value `365243` in `DAYS_EMPLOYED` (18% of rows) is not a real number of days; it is a code for "not employed/pensioner", so we turn it into NaN plus a flag.

## 9. Feature engineering and secondary tables
* **Ratios** capture affordability: credit/income, annuity/income, credit/goods price, implied loan term, employed years / age.
* **EXT_SOURCE_1/2/3** are anonymised external credit scores: the strongest single predictors; combinations (mean, min, product) help.
* **Secondary tables** hold the applicant's *history before this application*: credit bureau records, previous Home Credit applications, instalment payments (late or short?), POS/cash loans and credit-card balances. We **aggregate** them to one row per applicant (mean, max, min, sum, counts) and join them to the application table.

## 10. Hyper-parameter tuning with Optuna
Hyper-parameters (number of leaves, regularisation, sampling rates...) are settings you choose, not learn. **Optuna** runs many trials, using the results of earlier trials to propose better ones (Bayesian optimisation, TPE). Always tune using CV on training data only, never the hold-out.

## 11. Interpretation
* **Feature importance (gain)**: how much each feature improved the loss across all trees. Biased toward high-cardinality / continuous features.
* **SHAP values** split each individual prediction fairly among the features: positive pushes risk up, negative pushes it down. The summary plot shows which features matter and in which direction. Importance is not causation.

## 12. Calibration
A score of 0.30 should mean "30% of such applicants default". **Calibration** checks this with a reliability diagram; **isotonic regression** can fix a distorted scale without changing the ranking. Tricks for imbalance (class weights, undersampling) improve recall at a threshold but *distort* probabilities.

## 13. Business use: thresholds, lift
* **Lift / gains table**: sort applicants by risk, cut into 10 deciles, and look at the default rate and share of all defaulters in each. "The riskiest 10% contain X% of defaulters".
* **Cost-based threshold**: if missing a defaulter costs 5x more than refusing a good customer, pick the threshold that minimises total cost.

## 14. Code map
`src/data.py` (loading, cleaning) -> `src/features.py` (ratios + table aggregations) -> `src/models.py` (model factories, CV with honest early stopping) -> `src/evaluation.py` (bootstrap AUC, thresholds, lift, calibration, segments) -> `scripts/01..05`.

## 15. Exercises
* Remove `EXT_SOURCE_*`: how far does AUC fall? What does that say about dependence on external scores?
* Add a target-encoded occupation feature *without* nested CV and watch the CV score inflate (leakage in action).
* Compare early stopping on the scoring fold vs on a slice of the training fold: how optimistic is the former?
* Change the threshold rule to flag the top 5% and compute the profit/loss with made-up costs.
