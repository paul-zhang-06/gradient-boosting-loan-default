"""Logistic regression vs random forest vs LightGBM vs XGBoost vs CatBoost on feature set B (application + engineered ratios).

5-fold stratified CV on 80% of the data (OOF AUC, per-fold AUC) and a single evaluation on the untouched 20% hold-out.
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models import Spec, cross_validate, fit_predict, holdout_split, load_matrix

TAB, CACHE = Path("outputs/tables"), Path("outputs/cache")
TAB.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    X, y, Xt, ids = load_matrix("B")
    tr, te = holdout_split(y)
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    Xho, yho = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    np.save(CACHE / "holdout_idx.npy", te)
    np.save(CACHE / "train_idx.npy", tr)
    specs = [Spec("Logistic regression", "lr"), Spec("Random forest", "rf"), Spec("LightGBM", "lgbm"), Spec("XGBoost", "xgb"), Spec("CatBoost", "cat")]
    rows, folds = [], []
    for spec in specs:
        t0 = time.time()
        print(f"== {spec.name}", flush=True)
        cv = cross_validate(spec, Xtr, ytr)
        np.save(CACHE / f"oof_B_{spec.kind}.npy", cv["oof"])
        p, _, info = fit_predict(spec, Xtr, ytr, Xho)
        np.save(CACHE / f"holdout_B_{spec.kind}.npy", p)
        auc_ho = roc_auc_score(yho, p)
        rows.append(dict(model=spec.name, cv_auc_mean=cv["cv_auc_mean"], cv_auc_std=cv["cv_auc_std"], oof_auc=cv["oof_auc"], holdout_auc=auc_ho,
                         fit_seconds_per_fold=cv["fit_s_mean"], best_iter=info.get("best_iter")))
        folds.append(cv["folds"].assign(model=spec.name))
        print(f"   CV {cv['cv_auc_mean']:.4f} +/- {cv['cv_auc_std']:.4f}   hold-out {auc_ho:.4f}   total {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "10_model_comparison_set_B.csv", index=False)
        pd.concat(folds).to_csv(TAB / "11_cv_fold_auc_set_B.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
