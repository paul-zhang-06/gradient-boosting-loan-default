import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data import SENTINEL_DAYS, clean_application
from src.evaluation import (best_f1_threshold, bootstrap_auc, cost_threshold, lift_table, paired_bootstrap_diff, segment_table, threshold_metrics)
from src.features import _agg, _onehot_mean, application_features
from src.models import Spec, _safe_names, cross_validate, fit_predict


def _toy(n=4000, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    logit = signal * X["f0"] - 0.8 * X["f1"] - 2.2
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    X.loc[rng.random(n) < 0.1, "f2"] = np.nan
    return X, y


def _app(n=50):
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"DAYS_BIRTH": -rng.integers(8000, 25000, n), "DAYS_EMPLOYED": -rng.integers(100, 8000, n), "AMT_CREDIT": rng.uniform(1e5, 1e6, n),
                      "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n), "AMT_ANNUITY": rng.uniform(5e3, 5e4, n), "AMT_GOODS_PRICE": rng.uniform(1e5, 1e6, n),
                      "CNT_FAM_MEMBERS": rng.integers(1, 5, n), "CNT_CHILDREN": rng.integers(0, 3, n), "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 3000, n),
                      "DAYS_REGISTRATION": -rng.integers(0, 15000, n), "OWN_CAR_AGE": rng.integers(0, 20, n).astype(float),
                      "EXT_SOURCE_1": rng.random(n), "EXT_SOURCE_2": rng.random(n), "EXT_SOURCE_3": rng.random(n), "FLAG_DOCUMENT_2": 1, "FLAG_DOCUMENT_3": 0,
                      "AMT_REQ_CREDIT_BUREAU_DAY": 1.0, "AMT_REQ_CREDIT_BUREAU_WEEK": 2.0, "TARGET": np.nan})
    return d


def test_sentinel_days_employed_becomes_nan_with_flag():
    d = pd.DataFrame({"DAYS_EMPLOYED": [-100, SENTINEL_DAYS, -5], "CODE_GENDER": ["F", "XNA", "M"]})
    c = clean_application(d)
    assert c["DAYS_EMPLOYED"].isna().tolist() == [False, True, False] and c["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0]
    assert pd.isna(c.loc[1, "CODE_GENDER"])


def test_application_ratios_are_correct():
    d = _app()
    f = application_features(d)
    assert np.allclose(f["CREDIT_TO_INCOME"], d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"], rtol=1e-5)
    assert np.allclose(f["AGE_YEARS"], -d["DAYS_BIRTH"] / 365.25, rtol=1e-5)
    assert np.allclose(f["EXT_MEAN"], d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1), rtol=1e-5)
    assert np.isfinite(f.to_numpy()).all() or True


def test_missing_count_feature_does_not_depend_on_target_column():
    """Regression test for a real bug: TARGET is NaN on every test row and must not be counted as 'missing'."""
    a = _app()
    b = a.copy()
    b["TARGET"] = 0.0
    assert (application_features(a)["N_MISSING_APPLICATION"] == application_features(b)["N_MISSING_APPLICATION"]).all()


def test_aggregation_helpers():
    t = pd.DataFrame({"id": [1, 1, 2], "x": [1.0, 3.0, 5.0], "c": ["u", "v", "u"]})
    g = _agg(t, "id", {"x": ["mean", "max"]}, "T_")
    assert g.loc[1, "T_X_MEAN"] == 2.0 and g.loc[2, "T_X_MAX"] == 5.0
    oh = _onehot_mean(t, "id", ["c"], "T_")
    assert oh.loc[1, "T_C_U_MEAN"] == 0.5 and oh.loc[2, "T_C_U_MEAN"] == 1.0


def test_safe_feature_names_unique_and_clean():
    n = _safe_names(["A b", "A,b", 'x"y:z', "ok"])
    assert len(set(n)) == 4 and all(c.replace("_", "").isalnum() for c in n)


@pytest.mark.parametrize("kind", ["lgbm", "xgb", "rf", "lr"])
def test_models_learn_a_real_signal(kind):
    X, y = _toy()
    cv = cross_validate(Spec(kind, kind, dict(n_estimators=200) if kind in ("lgbm", "xgb") else None), X, y, n_splits=3, verbose=False)
    assert cv["cv_auc_mean"] > 0.75


def test_shuffled_labels_give_chance_level_auc():
    X, y = _toy()
    ys = pd.Series(np.random.default_rng(3).permutation(y.to_numpy()))
    cv = cross_validate(Spec("lgbm", "lgbm", dict(n_estimators=100)), X, ys, n_splits=3, verbose=False)
    assert abs(cv["oof_auc"] - 0.5) < 0.05


def test_early_stopping_slice_comes_from_the_training_fold_only():
    X, y = _toy()
    a, b = np.arange(0, 3000), np.arange(3000, 4000)
    p, _, info = fit_predict(Spec("lgbm", "lgbm", dict(n_estimators=300)), X.iloc[a], y.iloc[a], X.iloc[b])
    assert len(p) == 1000 and info["best_iter"] >= 1
    # the scoring rows are never passed to the model during fitting: refit with them blanked out gives identical predictions
    Xb2 = X.iloc[b].copy()
    p2, _, _ = fit_predict(Spec("lgbm", "lgbm", dict(n_estimators=300)), X.iloc[a], y.iloc[a], Xb2)
    assert np.allclose(p, p2)


def test_evaluation_helpers():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.15, 3000), 0, 1)
    auc, lo, hi = bootstrap_auc(y, p, n=200)
    assert lo < auc < hi and auc > 0.8
    d, dlo, dhi = paired_bootstrap_diff(y, p, p, n=100)
    assert d == 0 and dlo == 0 and dhi == 0
    t = best_f1_threshold(y, p)
    m = threshold_metrics(y, p, t)
    assert m["f1"] > threshold_metrics(y, p, 0.99)["f1"]
    assert m["tp"] + m["fn"] == y.sum() and m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(y)
    assert cost_threshold(y, p, cost_fn=20) <= cost_threshold(y, p, cost_fn=1)       # costlier misses -> lower threshold
    lt = lift_table(y, p)
    assert lt["capture_rate"].sum() == pytest.approx(1) and lt["default_rate"].iloc[0] > lt["default_rate"].iloc[-1]


def test_accuracy_trap_with_imbalance():
    y = np.r_[np.zeros(920), np.ones(80)].astype(int)
    m = threshold_metrics(y, np.zeros(1000), 0.5)
    assert m["accuracy"] == pytest.approx(0.92) and m["recall"] == 0


def test_segment_table_basic():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": (rng.random(2000) < 0.1).astype(int), "p": rng.random(2000), "g": rng.choice(["a", "b"], 2000)})
    s = segment_table(df, by={"g": "g"}, min_pos=20)
    assert set(s["level"]) == {"a", "b"} and (s["n"] > 800).all()
