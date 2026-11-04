"""Sensitive-attribute and key-feature ablations: what does the model lose without gender / age proxies / external scores, and who gets flagged?"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation import bootstrap_auc
from src.models import Spec, fit_predict, holdout_split, load_matrix

TAB = Path("outputs/tables")
bp = json.loads(Path("outputs/best_params.json").read_text())
TUNED = dict(bp["params"], learning_rate=0.03)
AGE_PROXIES = ["DAYS_BIRTH", "AGE_YEARS", "EMPLOYED_TO_AGE", "PHONE_CHANGE_TO_AGE", "REGISTRATION_TO_AGE", "CAR_AGE_TO_AGE"]

if __name__ == "__main__":
    X, y, _, _ = load_matrix("C")
    tr, te = holdout_split(y)
    Xtr, ytr, Xho, yho = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    ext = [c for c in X.columns if c.startswith("EXT_")]
    variants = {"full model": [], "without CODE_GENDER": ["CODE_GENDER"], "without gender and age proxies": ["CODE_GENDER"] + AGE_PROXIES,
                "without external scores (EXT_*)": ext}
    rows, preds = [], {}
    for nm, drop in variants.items():
        cols = [c for c in X.columns if c not in drop]
        p = np.load("outputs/cache/holdout_C_tuned.npy") if nm == "full model" else fit_predict(Spec(nm, "lgbm", TUNED), Xtr[cols], ytr, Xho[cols])[0]
        preds[nm] = p
        a, lo, hi = bootstrap_auc(yho, p, n=300)
        rows.append(dict(variant=nm, n_features=len(cols), holdout_auc=a, ci_low=lo, ci_high=hi))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(TAB / "40_ablation_sensitive_and_key_features.csv", index=False)
    # group metrics by gender: who is flagged and who is caught
    g = Xho["CODE_GENDER"].astype(str)
    out = []
    for nm, p in preds.items():
        thr = np.quantile(p, 0.8)
        for lvl in ["F", "M"]:
            m = (g == lvl).to_numpy()
            out.append(dict(variant=nm, gender=lvl, n=int(m.sum()), actual_default_rate=float(yho[m].mean()), mean_predicted=float(p[m].mean()),
                            flagged_share_top20=float((p[m] >= thr).mean()), recall_top20=float(((p[m] >= thr) & (yho[m] == 1)).sum() / yho[m].sum()),
                            auc=float(roc_auc_score(yho[m], p[m]))))
    pd.DataFrame(out).to_csv(TAB / "41_gender_group_metrics.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string())
