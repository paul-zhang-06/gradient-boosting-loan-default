"""Does adding engineered ratios and each secondary table improve AUC? LightGBM, same folds, 5-fold CV on the 80% training split."""
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
GROUPS = {"bureau": "BUREAU_", "previous_application": "PREV_", "installments_payments": "INST_", "POS_CASH_balance": "POS_", "credit_card_balance": "CARD_"}

if __name__ == "__main__":
    XC, y, _, _ = load_matrix("C")
    XB, _, _, _ = load_matrix("B")
    XA, _, _, _ = load_matrix("A")
    tr, te = holdout_split(y)
    extra = [c for c in XC.columns if c not in XB.columns]
    cols_by_group = {g: [c for c in extra if c.startswith(p)] for g, p in GROUPS.items()}
    assert sum(len(v) for v in cols_by_group.values()) == len(extra), "unassigned secondary-table columns"
    sets = {"A: application (raw)": list(XA.columns), "B: + engineered ratios": list(XB.columns)}
    for g, cs in cols_by_group.items():
        sets[f"B + {g}"] = list(XB.columns) + cs
    sets["C: B + all secondary tables"] = list(XC.columns)
    ytr, yho = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    rows = []
    for name, cols in sets.items():
        t0 = time.time()
        Xtr = XC.iloc[tr][cols].reset_index(drop=True)
        cv = cross_validate(Spec("lgbm", "lgbm"), Xtr, ytr, verbose=False)
        rows.append(dict(feature_set=name, n_features=len(cols), cv_auc_mean=cv["cv_auc_mean"], cv_auc_std=cv["cv_auc_std"], oof_auc=cv["oof_auc"],
                         fit_seconds_per_fold=cv["fit_s_mean"]))
        print(f"{name:34s} n={len(cols):4d}  CV {cv['cv_auc_mean']:.4f} +/- {cv['cv_auc_std']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "12_feature_ablation_lightgbm.csv", index=False)
    d = pd.DataFrame(rows)
    d["gain_vs_B"] = d["cv_auc_mean"] - float(d.loc[d.feature_set.str.startswith("B:"), "cv_auc_mean"].iloc[0])
    d.to_csv(TAB / "12_feature_ablation_lightgbm.csv", index=False)
    print(d.round(4).to_string())
