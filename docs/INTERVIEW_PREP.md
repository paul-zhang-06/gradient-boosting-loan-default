# Interview preparation: gradient boosting for loan default

(See the handbook PDF for the full 29-question bank.)

**Pitch.** "I built a leak-checked pipeline on Home Credit Default Risk. On identical application features LightGBM/XGBoost/CatBoost reach about 0.77 hold-out AUC versus 0.75 for logistic regression and random forest. The biggest gain, +0.025 CV AUC, came from aggregating the six history tables; tuning added only 0.002.
Final tuned LightGBM: hold-out AUC 0.7955 (CI 0.789-0.802). I checked leakage with a shuffled-target test (0.498), an ID test and adversarial validation, which showed the official test set is shifted from training (AUC 0.89)."
**Why AUC? Why not accuracy?** 8% positives: everyone-repays is 91.9% accurate. AUC is the probability a random defaulter outranks a random non-defaulter; threshold-free. Use PR curve, recall/precision at a business threshold, lift.
**Why did boosting beat logistic regression?** Interactions, non-linearities, native missing values; +0.017 AUC. Logistic regression is the transparent baseline.
**Early stopping and leakage?** Stopping on the scoring fold inflates CV; I carve a slice from each training fold.
**How did you pick the threshold?** On out-of-fold predictions, not the hold-out: max-F1 0.168 (flags 12.4%, recall 0.45, precision 0.29) or a cost rule with an assumed 5:1 miss/false-alarm cost. It is a business decision.
**Class weights/undersampling?** Hurt: AUC 0.7625 to 0.7166 (weights); predictions inflated to 10% / 22% mean risk; best F1 not better. Ranking metric AUC and calibrated probabilities were better without them.
**Calibration?** Already good (Brier 0.0649, mean 8.02% vs 8.07%); isotonic did not help.
**Is the model stable?** CV sd 0.002; tuned beats default in 5/5 folds; three seeds 0.7947-0.7959; 18 of top-20 features in all folds.
**Leakage story?** The adversarial test caught my own bug: the missing-count feature included the empty TARGET column. Fixed; unit test added. It also showed true shift: test is 99.1% cash loans vs 90.5%.
**SHAP vs gain?** Gain is biased towards continuous high-cardinality features; SHAP gives direction and per-applicant explanations; top-10 overlap is 6 of 10. Neither is causal.
**Fairness?** Gender is the 2nd SHAP feature; without gender and age proxies AUC falls only 0.003. A lender would drop them; residual proxies mean flag-rate gaps persist (27% vs 16% to 25% vs 17%).
**Biggest weaknesses?** No time variable (random split), distribution shift to test, tuning on a subsample, cost ratio assumed, no leaderboard score.
**Next steps?** Out-of-time validation, monotonic constraints, stacking, profit-based threshold, population-stability monitoring, fairness-constrained training.
