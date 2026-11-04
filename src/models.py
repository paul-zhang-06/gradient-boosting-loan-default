"""Model factories and an honest cross-validation routine.

Protocol
  * hold-out: 20% of application_train, stratified, touched ONCE at the end (out-of-sample AUC);
  * CV: 5-fold stratified on the remaining 80% -> out-of-fold (OOF) predictions and per-fold AUC;
  * early stopping uses a validation slice carved out of each TRAINING fold, never the scoring fold, so the CV AUC is not optimistic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42


# ------------------------------------------------------------------------------------------------ data assembly
def load_matrix(feature_set: str, processed="data/processed") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, test_ids). Categoricals become pandas 'category' (categories fit on train+test).

    feature_set: 'A' application only | 'B' + engineered ratios | 'C' + all secondary tables.
    """
    app = pd.read_parquet(f"{processed}/application.parquet")
    ids = app["SK_ID_CURR"]
    y = app["TARGET"]
    engineered = [c for c in app.columns if c in ENGINEERED]
    drop = ["SK_ID_CURR", "TARGET"] + ([] if feature_set != "A" else engineered)
    X = app.drop(columns=drop)
    if feature_set == "C":
        for name in ["bureau", "previous", "installments", "pos", "card"]:
            X = X.join(pd.read_parquet(f"{processed}/{name}.parquet"), on=app["SK_ID_CURR"]) if False else \
                pd.concat([X, pd.read_parquet(f"{processed}/{name}.parquet").reindex(ids.to_numpy()).reset_index(drop=True)], axis=1)
    X.columns = _safe_names(X.columns)
    for c in X.columns:                                   # division by zero in ratio features -> +/-inf: treat as missing
        if pd.api.types.is_float_dtype(X[c]):
            v = X[c].to_numpy()
            if np.isinf(v).any():
                X[c] = np.where(np.isinf(v), np.nan, v).astype(np.float32)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    is_train = y.notna().to_numpy()
    return X[is_train].reset_index(drop=True), y[is_train].astype(int).reset_index(drop=True), X[~is_train].reset_index(drop=True), ids[~is_train].reset_index(drop=True)


def _safe_names(cols) -> list[str]:
    """LightGBM forbids JSON characters in feature names (one-hot names contain commas, quotes, colons...)."""
    import re
    out, seen = [], {}
    for c in cols:
        n = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


ENGINEERED = ["AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYED_TO_AGE", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TERM_MONTHS", "CREDIT_TO_GOODS",
              "INCOME_PER_PERSON", "INCOME_PER_CHILD", "PHONE_CHANGE_TO_AGE", "REGISTRATION_TO_AGE", "CAR_AGE_TO_AGE", "EXT_MEAN", "EXT_MIN",
              "EXT_MAX", "EXT_STD", "EXT_PROD", "EXT_WEIGHTED", "EXT_N_MISSING", "DOCUMENTS_PROVIDED", "N_ENQUIRIES_TOTAL", "N_MISSING_APPLICATION"]


def holdout_split(y: pd.Series, test_size=0.2) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=SEED)


# ------------------------------------------------------------------------------------------------ model wrappers
@dataclass
class Spec:
    name: str
    kind: str                 # lgbm | xgb | cat | rf | lr
    params: dict | None = None


def _ordinal(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes.astype(np.float32).replace(-1, np.nan)
    return X


def fit_predict(spec: Spec, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, Xother: list[pd.DataFrame] = (), es_frac=0.1,
                seed=SEED, n_jobs=4) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Fit on (Xtr, ytr) with early stopping on a slice of Xtr; return predictions for Xva and every frame in Xother."""
    p = dict(spec.params or {})
    info = {}
    t0 = time.time()
    if spec.kind in ("lgbm", "xgb", "cat"):
        i_fit, i_es = train_test_split(np.arange(len(ytr)), test_size=es_frac, stratify=ytr, random_state=seed)
        Xf, yf, Xe, ye = Xtr.iloc[i_fit], ytr.iloc[i_fit], Xtr.iloc[i_es], ytr.iloc[i_es]
    if spec.kind == "lgbm":
        import lightgbm as lgb
        base = dict(n_estimators=5000, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.4, reg_alpha=0.1, reg_lambda=1.0, n_jobs=n_jobs, random_state=seed, verbosity=-1)
        base.update(p)
        m = lgb.LGBMClassifier(**base)
        m.fit(Xf, yf, eval_set=[(Xe, ye)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        info["best_iter"] = int(m.best_iteration_)
        pred = lambda X: m.predict_proba(X, num_iteration=m.best_iteration_)[:, 1]
    elif spec.kind == "xgb":
        import xgboost as xgb
        base = dict(n_estimators=5000, learning_rate=0.05, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0,
                    tree_method="hist", enable_categorical=True, eval_metric="auc", early_stopping_rounds=100, n_jobs=n_jobs, random_state=seed,
                    max_cat_to_onehot=1)
        base.update(p)
        m = xgb.XGBClassifier(**base)
        m.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
        info["best_iter"] = int(m.best_iteration)
        pred = lambda X: m.predict_proba(X, iteration_range=(0, m.best_iteration + 1))[:, 1]
    elif spec.kind == "cat":
        from catboost import CatBoostClassifier
        cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
        conv = lambda X: X.assign(**{c: X[c].astype(object).where(X[c].notna(), "missing").astype(str) for c in cats})   # CatBoost: NaN -> "missing"
        base = dict(iterations=3000, learning_rate=0.1, depth=6, eval_metric="AUC", early_stopping_rounds=100, thread_count=n_jobs,
                    random_seed=seed, verbose=0, cat_features=cats, l2_leaf_reg=5)
        base.update(p)
        m = CatBoostClassifier(**base)
        m.fit(conv(Xf), yf, eval_set=(conv(Xe), ye))
        info["best_iter"] = int(m.get_best_iteration())
        pred = lambda X: m.predict_proba(conv(X))[:, 1]
    elif spec.kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        base = dict(n_estimators=300, min_samples_leaf=40, max_features=0.3, n_jobs=n_jobs, random_state=seed, class_weight="balanced_subsample")
        base.update(p)
        imp = SimpleImputer(strategy="median")
        m = RandomForestClassifier(**base)
        Xo = _ordinal(Xtr)
        cols = Xo.columns[Xo.notna().any()]
        m.fit(imp.fit_transform(Xo[cols]), ytr)
        pred = lambda X: m.predict_proba(imp.transform(_ordinal(X)[cols]))[:, 1]
    elif spec.kind == "lr":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
        nums = [c for c in Xtr.columns if c not in cats and Xtr[c].notna().any()]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=False)), ("sc", StandardScaler())]), nums),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=100))]), cats)])
        base = dict(C=p.pop("C", 0.05), max_iter=300, solver="lbfgs")
        m = Pipeline([("pre", pre), ("lr", LogisticRegression(**base))])
        conv = lambda X: X.assign(**{c: X[c].astype(object) for c in cats})
        m.fit(conv(Xtr), ytr)
        pred = lambda X: m.predict_proba(conv(X))[:, 1]
    else:
        raise ValueError(spec.kind)
    info["fit_s"] = time.time() - t0
    info["model"] = m
    return pred(Xva), [pred(X) for X in Xother], info


def cross_validate(spec: Spec, X: pd.DataFrame, y: pd.Series, n_splits=5, seed=SEED, n_jobs=4, verbose=True) -> dict:
    """Stratified K-fold. Returns OOF predictions, per-fold AUC, best iterations and timing."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    for k, (a, b) in enumerate(skf.split(X, y), start=1):
        p, _, info = fit_predict(spec, X.iloc[a], y.iloc[a], X.iloc[b], seed=seed, n_jobs=n_jobs)
        oof[b] = p
        rows.append(dict(fold=k, auc=roc_auc_score(y.iloc[b], p), best_iter=info.get("best_iter"), fit_s=info["fit_s"]))
        if verbose:
            print(f"  {spec.name} fold {k}: AUC {rows[-1]['auc']:.5f}  iters {rows[-1]['best_iter']}  {rows[-1]['fit_s']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    return dict(name=spec.name, oof=oof, folds=df, cv_auc_mean=float(df["auc"].mean()), cv_auc_std=float(df["auc"].std()),
                oof_auc=float(roc_auc_score(y, oof)), fit_s_mean=float(df["fit_s"].mean()))
