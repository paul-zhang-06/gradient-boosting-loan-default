# Gradient Boosting for Loan Default (Home Credit Default Risk)

A complete, leak-checked machine-learning pipeline predicting loan default on the Kaggle **Home Credit Default Risk** data (307,511 applicants, 8.07% defaults, 122 columns + 6 related tables).
It compares logistic regression, random forest, LightGBM, XGBoost and CatBoost, measures what each extra table adds, tunes LightGBM with Optuna, and evaluates the final model **once** on an untouched hold-out with confidence intervals, SHAP, calibration, threshold analysis and error analysis.

> **Headline:** tuned LightGBM on all tables reaches **hold-out AUC 0.7955 (95% CI 0.789-0.802)**, five-fold CV AUC 0.7929 +/- 0.0018. Logistic regression on the same application features scores 0.752. The biggest single improvement comes from **the secondary tables (+0.025 AUC)**, not from model choice (the three boosting libraries differ by 0.001) or tuning (+0.002).

## Results at a glance
| Model | Features | 5-fold CV AUC | Hold-out AUC |
|---|---|---|---|
| Logistic regression | B: application + engineered ratios | 0.7492 +/- 0.0030 | 0.7517 |
| Random forest | B | 0.7482 +/- 0.0024 | 0.7521 |
| LightGBM | B | 0.7648 +/- 0.0020 | 0.7697 |
| XGBoost | B | 0.7636 +/- 0.0016 | 0.7699 |
| CatBoost | B | 0.7646 +/- 0.0019 | 0.7708 |
| LightGBM, default | C: B + all 5 secondary tables (688 features) | 0.7895 +/- 0.0025 | 0.7936 |
| **LightGBM, Optuna-tuned** | C | **0.7929 +/- 0.0018** | **0.7955** [0.789, 0.802] |

Tuned minus default on the hold-out: +0.0019 (paired-bootstrap 95% CI +0.0003 to +0.0034); tuned wins in 5 of 5 CV folds. Gini 0.59, KS 0.45. Three seeds: 0.7947-0.7959.

### What each table adds (LightGBM, 5-fold CV, `outputs/tables/12_*.csv`)
| Feature set | CV AUC | vs B |
|---|---|---|
| A: application, raw | 0.7574 | -0.0073 |
| B: + engineered ratios (credit/income, annuity/income, implied term, EXT_SOURCE mean/min/max/product ...) | 0.7648 | - |
| B + bureau (+ bureau_balance) | 0.7744 | +0.0096 |
| B + previous_application | 0.7751 | +0.0103 |
| B + installments_payments | 0.7754 | +0.0106 |
| B + POS_CASH_balance | 0.7721 | +0.0073 |
| B + credit_card_balance | 0.7687 | +0.0039 |
| **C: B + all tables** | **0.7895** | **+0.0247** |
Individual gains add up to 0.042 but the combined gain is 0.025: the tables carry overlapping information.

## Leakage and data-quality checks (`scripts/01_*.py`)
| Check | Result |
|---|---|
| Single-feature AUC scan (688 features) | strongest are the external scores (EXT_SOURCE_3 0.68; engineered EXT_WEIGHTED 0.74); **nothing near 0.9**; 8 features above 0.65 |
| AUC of `SK_ID_CURR` | 0.498 (no ordering leak) |
| **Shuffled-target test** (permute labels, CV) | **0.498**: the pipeline has no leak |
| Early stopping | uses a slice of each *training* fold, never the scoring fold |
| Secondary tables | only history *before* the application; unsupervised aggregations; no target encoding anywhere |
| **Adversarial validation, train vs test** | AUC **0.89** on raw application features: the official test set is **not a random sample of train** (99.1% cash loans vs 90.5%; different bureau-enquiry mix; mean credit 517k vs 599k). Expect CV and leaderboard scores to differ |
| A bug the checks caught | my "missing-value count" feature included the empty `TARGET` column, so every test row had +1 (adversarial AUC 0.99). Fixed and covered by a unit test |

## Class imbalance, thresholds, calibration (hold-out, tuned model)
* Predicting "everyone repays" is **91.9% accurate** and useless. At threshold 0.5 the model has accuracy 92.1% but recall **5.6%**.
* Thresholds chosen on out-of-fold predictions: max-F1 (0.168) flags 12.4%, precision 0.29, recall 0.45, F1 0.35; a 5:1 miss-vs-false-alarm cost rule gives almost the same; flagging the top 10% gives precision 0.32, recall 0.40.
* **Lift:** the riskiest decile has a 31.8% default rate (3.9x average) and contains 39.5% of all defaulters; the top two deciles contain 58.5%; the safest decile 0.8%.
* **Calibration is already good** (mean predicted 8.02% vs actual 8.07%; Brier 0.0649): isotonic calibration does not help.
* **Rebalancing tricks hurt** (LightGBM, set B, 3-fold): class weights cut AUC from 0.7625 to 0.7166 and inflate mean predicted risk to 10.1%; undersampling 3:1 gives a mean prediction of 21.9% (badly miscalibrated); neither improves the best F1 (0.31 baseline vs 0.27 / 0.31).

## Interpretation (`outputs/figures/12-15_*.png`)
* The three anonymous **external scores dominate**: EXT_MEAN is the top feature by gain and SHAP; removing all EXT_* features drops hold-out AUC by 0.021 (to 0.775), the secondary tables partly compensate.
* By source, share of total gain: engineered application features 30%, raw application 22%, previous_application 17%, bureau 14%, installments 10%, credit card 4%, POS 3%.
* Importance is stable: 18 of the top-20 features appear in the top 20 of **all five** CV folds (mean rank correlation 0.96).
* SHAP: high external scores, longer implied loan term and higher education reduce risk; high bureau debt-to-credit, higher credit-to-goods ratio, male gender, short employment raise it.
* **Fairness note:** `CODE_GENDER` is the 2nd most important SHAP feature. Without it hold-out AUC is 0.7941 (-0.0014); without gender and age-related features 0.7924 (-0.0031) - tiny losses, so a lender would likely drop them (regulatory exposure), at the cost of some residual proxy effects (flag-rate gap by gender narrows only from 27.2% vs 16.3% to 25.1% vs 17.4%).

## Error analysis (hold-out, `26_*`, `27_*`)
* Calibration holds inside every segment (mean prediction ~ actual default rate); AUC per segment 0.75-0.81 (worst: age 60+ 0.752, low-skill laborers 0.750).
* When the riskiest 20% are flagged, the share of defaulters caught is 66% for men vs 53% for women, 71% for laborers vs 39% for accountants, 60% for cash vs **35% for revolving loans**, **31% for age 60+**: low-base-rate groups are harder to separate.
* 14.3% of defaulters sit in the *safest half* of the ranking. They have higher external scores (median EXT_MEAN 0.59 vs 0.32 for caught defaulters), are older (48 vs 37) and have longer employment (5.2 vs 2.9 years): nothing in the data distinguishes them from repaying applicants.
* 60% of the riskiest 5% did **not** default (precision is limited by inherent uncertainty).
* Learning curve: hold-out AUC 0.772 (10% of the data) -> 0.784 -> 0.790 -> 0.796 (100%): still rising, more data would help.

## Kaggle submission
`outputs/submission.csv` (48,744 rows, probabilities) from a model refit on all labelled rows (1,491 trees). **It cannot be scored here** because the test labels are not public; upload it to Kaggle to get the leaderboard AUC. Mean predicted default rate on test is 7.3% (8.1% in train), consistent with the different test composition.

## Repository
`src/` (data, features, models, evaluation, plots) - `scripts/` 01 EDA + leakage, 02 model comparison, 02b CatBoost, 03 table ablation, 04 Optuna, 05 final evaluation, 06 fairness ablation - `tests/` 14 tests - `notebooks/` - `reports/report.md` - `docs/`.

```bash
python -m venv .venv && .venv\Scripts\activate         # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
# download the data from https://www.kaggle.com/c/home-credit-default-risk/data into data/raw/  (needs a Kaggle account)
python scripts/build_features.py       # ~1 min  -> data/processed
python scripts/01_eda_and_leakage_checks.py
python scripts/02_model_comparison.py ; python scripts/02b_catboost.py
python scripts/03_feature_ablation.py ; python scripts/04_tune_lightgbm.py
python scripts/05_final_evaluation.py ; python scripts/06_fairness_and_ablation.py
pytest
```
Total runtime about 1.5-2 hours on 4 CPU cores and 15 GB RAM.

## Limitations
* **Distribution shift:** the official test set differs from training (adversarial AUC 0.89): hold-out AUC is an estimate for the *training* population.
* **No time variable:** no temporal validation is possible, so the split is random; real deployment needs out-of-time testing and drift monitoring.
* Hyper-parameters tuned on a 100k-row subsample (transfer to full data is not guaranteed; the gain over defaults is small); XGBoost/CatBoost/RF not tuned (comparison used sensible defaults).
* Cost ratio (5:1) for thresholds is an assumption; no real profit model.
* Features that proxy protected attributes (gender, age) remain; fairness was examined, not enforced.
* Imbalance variants used default hyper-parameters, not re-tuned for weighted losses.
* Kaggle leaderboard score not obtained.

## Possible next steps
Out-of-time validation, target-encoded categorical features with nested CV, stacking of LightGBM/XGBoost/CatBoost, monotonic constraints on key features, fairness-constrained training, model-risk documentation (population-stability index), and a profit-based threshold.
