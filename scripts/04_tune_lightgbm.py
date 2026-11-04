"""Optuna tuning of LightGBM on feature set C. 3-fold CV on a 100k stratified subsample of the 80% training split (the hold-out is never used)."""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models import SEED, Spec, fit_predict, holdout_split, load_matrix
from src.plots import save

TAB = Path("outputs/tables")
N_TRIALS, SUBSAMPLE = 30, 100_000


def make_objective(X, y):
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))

    def objective(trial):
        params = dict(
            learning_rate=0.08,
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        )
        aucs, its = [], []
        for a, b in folds:
            p, _, info = fit_predict(Spec("lgbm", "lgbm", params), X.iloc[a], y.iloc[a], X.iloc[b])
            aucs.append(roc_auc_score(y.iloc[b], p))
            its.append(info["best_iter"])
        trial.set_user_attr("best_iter_mean", float(np.mean(its)))
        trial.set_user_attr("auc_std", float(np.std(aucs)))
        return float(np.mean(aucs))

    return objective


if __name__ == "__main__":
    X, y, _, _ = load_matrix("C")
    tr, te = holdout_split(y)
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    sub, _ = train_test_split(np.arange(len(ytr)), train_size=SUBSAMPLE, stratify=ytr, random_state=SEED)
    Xs, ys = Xtr.iloc[sub].reset_index(drop=True), ytr.iloc[sub].reset_index(drop=True)
    # reference: default parameters on the same subsample and folds, learning rate 0.08
    obj = make_objective(Xs, ys)
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(num_leaves=31, max_depth=12, min_child_samples=100, colsample_bytree=0.4, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0, min_split_gain=0.0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(obj, n_trials=N_TRIALS, timeout=3000, callbacks=[lambda s, t: print(f"trial {t.number:2d}  AUC {t.value:.5f}  best {s.best_value:.5f}  {time.time() - t0:.0f}s", flush=True)])
    df = study.trials_dataframe()
    df.to_csv(TAB / "13_optuna_trials.csv", index=False)
    Path("outputs/best_params.json").write_text(json.dumps(dict(params=study.best_params, best_auc_subsample_3fold=study.best_value,
                                                               default_auc_subsample_3fold=float(df.loc[0, "value"]),
                                                               best_iter_mean=study.best_trial.user_attrs["best_iter_mean"], n_trials=len(df)), indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(df["number"], df["value"], "o", alpha=.6, label="trial")
    axes[0].plot(df["number"], df["value"].cummax(), "r-", label="best so far")
    axes[0].axhline(df.loc[0, "value"], color="k", ls="--", lw=1, label="default parameters (trial 0)")
    axes[0].set(title="Optuna optimisation history (3-fold AUC, 100k subsample)", xlabel="trial", ylabel="AUC")
    axes[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp)[::-1], list(imp.values())[::-1])
        axes[1].set(title="Hyper-parameter importance")
    except Exception as e:
        axes[1].text(0.1, 0.5, str(e))
    save(fig, "07_optuna_history_and_importance.png")
    print(json.dumps(json.loads(Path("outputs/best_params.json").read_text()), indent=2))
