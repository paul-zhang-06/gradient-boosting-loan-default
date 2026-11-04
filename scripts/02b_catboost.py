"""CatBoost on feature set B, appended to the model comparison table (kept separate because it is the slowest model)."""
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

if __name__ == "__main__":
    X, y, _, _ = load_matrix("B")
    tr, te = holdout_split(y)
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    Xho, yho = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    spec = Spec("CatBoost", "cat")
    t0 = time.time()
    cv = cross_validate(spec, Xtr, ytr)
    np.save(CACHE / "oof_B_cat.npy", cv["oof"])
    p, _, info = fit_predict(spec, Xtr, ytr, Xho)
    np.save(CACHE / "holdout_B_cat.npy", p)
    row = dict(model="CatBoost", cv_auc_mean=cv["cv_auc_mean"], cv_auc_std=cv["cv_auc_std"], oof_auc=cv["oof_auc"], holdout_auc=roc_auc_score(yho, p),
               fit_seconds_per_fold=cv["fit_s_mean"], best_iter=info.get("best_iter"))
    f = TAB / "10_model_comparison_set_B.csv"
    d = pd.read_csv(f)
    d = pd.concat([d[d.model != "CatBoost"], pd.DataFrame([row])], ignore_index=True)
    d.to_csv(f, index=False)
    ff = TAB / "11_cv_fold_auc_set_B.csv"
    pd.concat([pd.read_csv(ff).query("model != 'CatBoost'"), cv["folds"].assign(model="CatBoost")]).to_csv(ff, index=False)
    print(d.round(4).to_string(), f"\ntotal {time.time() - t0:.0f}s")
