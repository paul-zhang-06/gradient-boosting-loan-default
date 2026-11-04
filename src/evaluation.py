"""Evaluation helpers: bootstrap AUC, threshold metrics, lift tables, calibration, segment analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)


def bootstrap_auc(y, p, n=1000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n):
        i = np.r_[rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))]
        out.append(roc_auc_score(y[i], p[i]))
    return float(roc_auc_score(y, p)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_bootstrap_diff(y, p1, p2, n=1000, seed=0) -> tuple[float, float, float]:
    """AUC(p1) - AUC(p2) with a paired bootstrap CI (same resampled rows for both models)."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def threshold_metrics(y, p, thr) -> dict:
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(threshold=float(thr), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), precision=float(precision_score(y, yhat, zero_division=0)),
                recall=float(recall_score(y, yhat)), f1=float(f1_score(y, yhat)), accuracy=float((tp + tn) / len(yhat)), flagged_share=float(yhat.mean()))


def best_f1_threshold(y, p, grid=None) -> float:
    grid = np.quantile(p, np.linspace(0.5, 0.999, 400)) if grid is None else grid
    f = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f))])


def cost_threshold(y, p, cost_fn=5.0, cost_fp=1.0, grid=None) -> float:
    """Threshold minimising  cost_fn * FN + cost_fp * FP  (an ASSUMED cost ratio: missing a defaulter costs `cost_fn` times a lost good customer)."""
    grid = np.quantile(p, np.linspace(0.3, 0.999, 500)) if grid is None else grid
    y = np.asarray(y)
    costs = [cost_fn * ((p < t) & (y == 1)).sum() + cost_fp * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def lift_table(y, p, n_bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    g = d.groupby("decile", observed=True).agg(n=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean"), mean_pred=("p", "mean")).reset_index()
    g["lift"] = g["default_rate"] / d["y"].mean()
    g["capture_rate"] = g["defaults"] / d["y"].sum()
    g["cum_capture_rate"] = g["capture_rate"].cumsum()
    return g


def ks_statistic(y, p) -> float:
    from scipy.stats import ks_2samp
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def calibration_summary(y, p) -> dict:
    return dict(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), mean_pred=float(np.mean(p)), base_rate=float(np.mean(y)),
                avg_precision=float(average_precision_score(y, p)))


def segment_table(df: pd.DataFrame, y_col="y", p_col="p", by=None, min_pos=40, flag_share=0.2) -> pd.DataFrame:
    """Per-segment size, default rate, mean predicted risk, AUC, and recall when the top `flag_share` riskiest applicants are flagged."""
    thr = df[p_col].quantile(1 - flag_share)
    rows = []
    for name, col in by.items():
        for lvl, g in df.groupby(col, observed=True, dropna=False):
            if len(g) < 200:
                continue
            pos = int(g[y_col].sum())
            auc = roc_auc_score(g[y_col], g[p_col]) if pos >= min_pos and pos < len(g) - min_pos else np.nan
            rows.append(dict(segment=name, level=str(lvl), n=len(g), default_rate=g[y_col].mean(), mean_pred=g[p_col].mean(), auc=auc,
                             recall_at_top20=float(((g[p_col] >= thr) & (g[y_col] == 1)).sum() / max(pos, 1))))
    return pd.DataFrame(rows)
