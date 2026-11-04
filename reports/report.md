# Report: gradient boosting for loan default

## 1. Question
How well can we rank applicants by default risk using the Home Credit data, which model family and which tables matter, is the evidence trustworthy (leakage, shift), and how should the scores be used (thresholds, calibration, fairness)?

## 2. Data
Home Credit Default Risk (Kaggle): application_train (307,511 x 122, 8.07% default), application_test (48,744), and six history tables (bureau, bureau_balance, previous_application, installments_payments, POS_CASH_balance, credit_card_balance). Quality issues: 41 columns above 50% missing; `DAYS_EMPLOYED = 365243` in 18% of rows (default rate 5.4% vs 8.7%); 4 rows with gender 'XNA'.

## 3. Method
* Feature sets: A application raw; B + 19 engineered ratios and an anomaly flag; C + aggregates of five secondary-table groups (688 features).
* Protocol: stratified 80/20 split; 5-fold stratified CV on the 80%; early stopping on a slice of each training fold; hold-out used once; Optuna (30 trials, 3-fold, 100k subsample) on training only.
* Evaluation: AUC with bootstrap CIs, PR curve, KS/Gini, thresholds from out-of-fold data, lift, calibration, SHAP, segment analysis, learning curve, seed variation.

## 4. Findings
1. Boosting beats linear/bagging by about 0.017 AUC on identical features; the three boosting libraries are within 0.001.
2. Feature engineering: +0.0074 (CV). Secondary tables: +0.0247 jointly (bureau, previous applications and instalments each about +0.010; credit card +0.004), sub-additive.
3. Tuning adds only +0.003 (CV) / +0.002 (hold-out), significant but small.
4. The pipeline passes the shuffled-target (0.498) and ID (0.498) tests; the adversarial test finds the official test set is shifted (0.89).
5. Ranking quality is good (top decile 3.9x lift, 39% of defaulters), and probabilities are already calibrated; imbalance tricks hurt.
6. External scores carry most of the signal (removing them: -0.021 AUC); gender and age proxies add little (-0.003).
7. Error analysis: calibration holds in every segment; missed defaulters look like repaying customers on the observed features.

## 5. Conclusion
A tuned LightGBM on all tables gives hold-out AUC 0.7955. Most of the achievable improvement came from adding relevant data, not from model sophistication. The result is an estimate for a population like the training data; the official test set is different, and no out-of-time validation was possible.

## 6. Limitations / next steps
See README.
