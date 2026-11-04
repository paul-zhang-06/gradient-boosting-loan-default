"""Final model, hold-out evaluation (once), thresholds, imbalance, calibration, SHAP, error analysis, learning curve, submission file."""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.evaluation import (bootstrap_auc, best_f1_threshold, calibration_summary, cost_threshold, ks_statistic, lift_table, paired_bootstrap_diff,
                            segment_table, threshold_metrics)
from src.models import SEED, Spec, fit_predict, holdout_split, load_matrix
from src.plots import save

TAB, CACHE = Path("outputs/tables"), Path("outputs/cache")
S = {}
bp = json.loads(Path("outputs/best_params.json").read_text())
TUNED = dict(bp["params"], learning_rate=0.03)
DEFAULT = dict()

X, y, Xtest, test_ids = load_matrix("C")
tr, te = holdout_split(y)
Xtr, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
Xho, yho = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
GROUPS = [("BUREAU_", "bureau + bureau_balance"), ("PREV_", "previous_application"), ("INST_", "installments_payments"), ("POS_", "POS_CASH_balance"), ("CARD_", "credit_card_balance")]


def group_of(c):
    for p, g in GROUPS:
        if c.startswith(p):
            return g
    return "application (engineered)" if c in set(load_matrix.__globals__["ENGINEERED"]) else "application (raw)"


# ------------------------------------------------------------------ 1. 5-fold CV of default vs tuned LightGBM (feature set C), keeping per-fold importance
t0 = time.time()
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = {"default": np.zeros(len(ytr)), "tuned": np.zeros(len(ytr))}
rows, imps, iters = [], [], []
for k, (a, b) in enumerate(skf.split(Xtr, ytr), start=1):
    for nm, prm in [("default", DEFAULT), ("tuned", TUNED)]:
        p, _, info = fit_predict(Spec(nm, "lgbm", prm), Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b])
        oof[nm][b] = p
        rows.append(dict(fold=k, model=nm, auc=roc_auc_score(ytr.iloc[b], p), best_iter=info["best_iter"], fit_s=info["fit_s"]))
        if nm == "tuned":
            iters.append(info["best_iter"])
            imps.append(pd.Series(info["model"].booster_.feature_importance("gain"), Xtr.columns))
    print(f"fold {k} done  {time.time() - t0:.0f}s  " + "  ".join(f"{r['model']} {r['auc']:.4f}" for r in rows[-2:]), flush=True)
folds = pd.DataFrame(rows)
folds.to_csv(TAB / "20_cv_folds_default_vs_tuned_set_C.csv", index=False)
np.save(CACHE / "oof_C_default.npy", oof["default"])
np.save(CACHE / "oof_C_tuned.npy", oof["tuned"])
cv = folds.groupby("model")["auc"].agg(["mean", "std"])
S["cv_auc_set_C"] = {m: dict(mean=float(cv.loc[m, "mean"]), std=float(cv.loc[m, "std"])) for m in cv.index}
fold_d = folds.pivot(index="fold", columns="model", values="auc")
S["tuned_beats_default_in_folds"] = int((fold_d["tuned"] > fold_d["default"]).sum())

# fold stability of feature importance (top-20 overlap)
imp_df = pd.concat(imps, axis=1)
imp_df.columns = [f"fold{i}" for i in range(1, 6)]
imp_norm = imp_df / imp_df.sum()
top_sets = [set(imp_norm[c].sort_values(ascending=False).head(20).index) for c in imp_norm.columns]
S["top20_importance_features_in_all_5_folds"] = int(len(set.intersection(*top_sets)))
S["top20_importance_rank_corr_mean"] = float(imp_norm.corr(method="spearman").to_numpy()[np.triu_indices(5, 1)].mean())

# ------------------------------------------------------------------ 2. HOLD-OUT (first and only evaluation): fit on the 80% training split
n_iter_final = int(np.mean(iters) * 1.0)
models = {}
ho = {}
for nm, prm in [("default", DEFAULT), ("tuned", TUNED)]:
    p, _, info = fit_predict(Spec(nm, "lgbm", prm), Xtr, ytr, Xho)
    ho[nm], models[nm] = p, info["model"]
    np.save(CACHE / f"holdout_C_{nm}.npy", p)
S["holdout_auc"] = {nm: bootstrap_auc(yho, ho[nm]) for nm in ho}
S["holdout_auc_diff_tuned_minus_default"] = paired_bootstrap_diff(yho, ho["tuned"], ho["default"])
S["gini_tuned"] = 2 * S["holdout_auc"]["tuned"][0] - 1
S["ks_tuned"] = ks_statistic(yho, ho["tuned"])
pB = {k: np.load(CACHE / f"holdout_B_{k}.npy") for k in ["lr", "rf", "lgbm", "xgb", "cat"] if (CACHE / f"holdout_B_{k}.npy").exists()}
S["holdout_auc_set_B"] = {k: roc_auc_score(yho, v) for k, v in pB.items()}
S["holdout_auc_diff_tuned_C_minus_lgbm_B"] = paired_bootstrap_diff(yho, ho["tuned"], pB["lgbm"])
S["holdout_auc_diff_tuned_C_minus_LR_B"] = paired_bootstrap_diff(yho, ho["tuned"], pB["lr"])

# ------------------------------------------------------------------ 3. curves
p_t, p_o = ho["tuned"], oof["tuned"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
names = {"lr": "Logistic regression (B)", "rf": "Random forest (B)", "lgbm": "LightGBM (B)", "xgb": "XGBoost (B)", "cat": "CatBoost (B)"}
for k, v in pB.items():
    f, t, _ = roc_curve(yho, v)
    axes[0].plot(f, t, lw=1.1, label=f"{names[k]}  AUC {roc_auc_score(yho, v):.3f}")
    pr, rc, _ = precision_recall_curve(yho, v)
    axes[1].plot(rc, pr, lw=1.1, label=f"{names[k]}  AP {np.trapezoid(pr[::-1], rc[::-1]):.3f}")
f, t, _ = roc_curve(yho, p_t)
axes[0].plot(f, t, "k", lw=2.2, label=f"Tuned LightGBM (C)  AUC {roc_auc_score(yho, p_t):.3f}")
pr, rc, _ = precision_recall_curve(yho, p_t)
axes[1].plot(rc, pr, "k", lw=2.2, label="Tuned LightGBM (C)")
axes[0].plot([0, 1], [0, 1], "grey", ls=":")
axes[0].set(title="ROC curves on the untouched hold-out (61,503 applicants)", xlabel="false positive rate", ylabel="true positive rate")
axes[1].axhline(yho.mean(), color="grey", ls=":", label=f"base rate {yho.mean():.3f}")
axes[1].set(title="Precision-recall curves (hold-out)", xlabel="recall", ylabel="precision")
for a in axes:
    a.legend(fontsize=8)
save(fig, "08_roc_and_pr_curves_holdout.png")

# ------------------------------------------------------------------ 4. thresholds (chosen on OOF = training data only), evaluated on hold-out
thr_f1 = best_f1_threshold(ytr, p_o)
thr_cost = cost_threshold(ytr, p_o, cost_fn=5.0, cost_fp=1.0)
thr_top10 = float(np.quantile(p_o, 0.90))
rows = []
for nm, thr in [("max-F1 (chosen on OOF)", thr_f1), ("cost-based FN:FP = 5:1 (chosen on OOF)", thr_cost), ("flag top 10% riskiest", thr_top10), ("0.5 (naive)", 0.5)]:
    rows.append(dict(rule=nm, **threshold_metrics(yho, p_t, thr)))
thr_tab = pd.DataFrame(rows)
thr_tab.to_csv(TAB / "21_threshold_rules_holdout.csv", index=False)
S["accuracy_of_predicting_everyone_repays"] = float(1 - yho.mean())
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, (nm, thr) in zip(axes, [("max-F1", thr_f1), ("cost 5:1", thr_cost), ("top 10% flagged", thr_top10)]):
    ConfusionMatrixDisplay.from_predictions(yho, (p_t >= thr).astype(int), ax=ax, cmap="Blues", colorbar=False, display_labels=["repaid", "default"])
    m = threshold_metrics(yho, p_t, thr)
    ax.set_title(f"{nm}: thr {thr:.3f}\nprecision {m['precision']:.2f}  recall {m['recall']:.2f}  F1 {m['f1']:.2f}")
save(fig, "09_confusion_matrices_holdout.png")
grid = np.quantile(p_o, np.linspace(0.5, 0.995, 120))
sw = pd.DataFrame([threshold_metrics(ytr, p_o, g) for g in grid])
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(sw["flagged_share"], sw["precision"], label="precision")
ax.plot(sw["flagged_share"], sw["recall"], label="recall")
ax.plot(sw["flagged_share"], sw["f1"], label="F1")
ax.set(xlabel="share of applicants flagged as risky", title="Threshold trade-off on out-of-fold predictions (class imbalance 8%)")
ax.legend()
save(fig, "10_threshold_tradeoff.png")

# ------------------------------------------------------------------ 5. lift / gains and calibration
lift = lift_table(yho, p_t)
lift.to_csv(TAB / "22_lift_table_holdout.csv", index=False)
S["top_decile_default_rate"], S["top_decile_capture"], S["top_2_deciles_capture"] = float(lift.default_rate[0]), float(lift.capture_rate[0]), float(lift.cum_capture_rate[1])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_o, ytr)
p_cal = iso.predict(p_t)
S["calibration_raw"], S["calibration_isotonic"] = calibration_summary(yho, p_t), calibration_summary(yho, p_cal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for lab, pp in [("raw", p_t), ("isotonic (fit on OOF)", p_cal)]:
    fr, mp = calibration_curve(yho, pp, n_bins=12, strategy="quantile")
    axes[0].plot(mp, fr, "o-", label=lab)
axes[0].plot([0, 0.6], [0, 0.6], "k:")
axes[0].set(title="Reliability diagram (hold-out)", xlabel="mean predicted default probability", ylabel="observed default rate")
axes[0].legend()
axes[1].bar(lift["decile"].astype(int), lift["default_rate"] * 100, color="C3")
axes[1].axhline(yho.mean() * 100, color="k", ls="--", lw=1)
for i, r in lift.iterrows():
    axes[1].text(r["decile"], r["default_rate"] * 100, f"{r['lift']:.1f}x", ha="center", va="bottom", fontsize=8)
axes[1].set(title="Default rate by predicted-risk decile (decile 1 = riskiest)", xlabel="risk decile", ylabel="% defaulted")
save(fig, "11_calibration_and_lift.png")

# ------------------------------------------------------------------ 6. importance and SHAP
imp_gain = pd.Series(models["tuned"].booster_.feature_importance("gain"), Xtr.columns)
imp_split = pd.Series(models["tuned"].booster_.feature_importance("split"), Xtr.columns)
imp_tab = pd.DataFrame({"gain_share": imp_gain / imp_gain.sum(), "split_share": imp_split / imp_split.sum()}).sort_values("gain_share", ascending=False)
imp_tab["source"] = [group_of(c) for c in imp_tab.index]
imp_tab.to_csv(TAB / "23_feature_importance_tuned.csv")
grp = imp_tab.groupby("source")["gain_share"].sum().sort_values()
S["importance_by_source_table"] = {k: float(v) for k, v in grp.items()}
S["n_features_used_gain_gt_0"] = int((imp_gain > 0).sum())
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
t = imp_tab.head(30).iloc[::-1]
axes[0].barh(t.index, t.gain_share, color=[sns.color_palette()[list(grp.index).index(s) % 10] for s in t.source])
axes[0].set(title="Top 30 features by gain (tuned LightGBM, set C)")
axes[0].tick_params(axis="y", labelsize=7)
axes[1].barh(grp.index, grp.values)
axes[1].set(title="Share of total gain by source table")
save(fig, "12_feature_importance.png")
imp_stab = pd.DataFrame({"mean_share": imp_norm.mean(axis=1), "std_share": imp_norm.std(axis=1)}).sort_values("mean_share", ascending=False).head(20)
imp_stab.to_csv(TAB / "24_importance_stability_across_folds.csv")

samp = yho.groupby(yho).apply(lambda s: s.sample(n=min(len(s), 5000 if s.name == 0 else 800), random_state=SEED)).index.get_level_values(1)
Xs = Xho.loc[samp]
contrib = models["tuned"].booster_.predict(Xs, pred_contrib=True)
sv, base = contrib[:, :-1], contrib[:, -1].mean()
Xs_num = Xs.copy()
for c in Xs_num.columns:
    if isinstance(Xs_num[c].dtype, pd.CategoricalDtype):
        Xs_num[c] = Xs_num[c].cat.codes.astype(float).replace(-1, np.nan)
shap_imp = pd.Series(np.abs(sv).mean(axis=0), Xs.columns).sort_values(ascending=False)
shap_imp.to_csv(TAB / "25_shap_mean_abs.csv", header=["mean_abs_shap"])
plt.figure(figsize=(10, 9))
shap.summary_plot(sv, Xs_num, max_display=25, show=False, plot_size=None)
plt.title("SHAP summary (tuned LightGBM, 5,800 hold-out applicants; log-odds)")
save(plt.gcf(), "13_shap_summary.png")
plt.figure(figsize=(8, 7))
shap.summary_plot(sv, Xs_num, plot_type="bar", max_display=20, show=False, plot_size=None)
save(plt.gcf(), "14_shap_bar.png")
top4 = list(shap_imp.index[:4])
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, f in zip(axes.ravel(), top4):
    xv = Xs_num[f].to_numpy()
    ax.scatter(xv, sv[:, list(Xs.columns).index(f)], s=5, alpha=.4, c=yho.loc[samp].to_numpy(), cmap="coolwarm")
    ax.axhline(0, color="k", lw=.6)
    ax.set(title=f"SHAP dependence: {f}", xlabel=f, ylabel="SHAP (log-odds)")
save(fig, "15_shap_dependence_top4.png")
S["shap_top10"] = [dict(feature=k, mean_abs_shap=float(v)) for k, v in shap_imp.head(10).items()]
S["gain_vs_shap_top10_overlap"] = int(len(set(imp_tab.index[:10]) & set(shap_imp.index[:10])))

# ------------------------------------------------------------------ 7. error analysis on the hold-out
df = pd.DataFrame({"y": yho.to_numpy(), "p": p_t}, index=Xho.index)
for c in ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "NAME_FAMILY_STATUS", "FLAG_OWN_REALTY"]:
    df[c] = Xho[c].astype(str)
df["AGE_BAND"] = pd.cut(Xho["AGE_YEARS"], [0, 30, 40, 50, 60, 100], labels=["<30", "30-40", "40-50", "50-60", "60+"])
df["INCOME_QUARTILE"] = pd.qcut(Xho["AMT_INCOME_TOTAL"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
df["EXT_SOURCES_MISSING"] = Xho["EXT_N_MISSING"].astype(int).astype(str)
df["HAS_BUREAU_HISTORY"] = np.where(Xho["BUREAU_COUNT"].notna(), "yes", "no")
df["HAS_PREVIOUS_APPLICATIONS"] = np.where(Xho["PREV_COUNT"].notna(), "yes", "no")
by = {c: c for c in ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "NAME_FAMILY_STATUS", "AGE_BAND", "INCOME_QUARTILE",
                     "EXT_SOURCES_MISSING", "HAS_BUREAU_HISTORY", "HAS_PREVIOUS_APPLICATIONS"]}
seg = segment_table(df, by=by)
seg.to_csv(TAB / "26_segment_error_analysis_holdout.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
s2 = seg[seg.segment.isin(["AGE_BAND", "INCOME_QUARTILE", "NAME_CONTRACT_TYPE", "HAS_BUREAU_HISTORY", "EXT_SOURCES_MISSING", "NAME_EDUCATION_TYPE"])].dropna(subset=["auc"])
s2 = s2.assign(label=s2.segment + ": " + s2.level).sort_values("auc")
axes[0].barh(s2.label, s2.auc)
axes[0].axvline(roc_auc_score(yho, p_t), color="r", ls="--", label="overall")
axes[0].set(title="AUC within customer segments (hold-out)", xlim=(0.6, 0.85))
axes[0].legend()
axes[0].tick_params(axis="y", labelsize=7)
s3 = seg[seg.segment.isin(["OCCUPATION_TYPE"])].sort_values("recall_at_top20")
axes[1].barh(s3.level, s3.recall_at_top20)
axes[1].set(title="Share of defaulters caught when the riskiest 20% are flagged, by occupation")
axes[1].tick_params(axis="y", labelsize=7)
save(fig, "16_segment_error_analysis.png")
# who is missed? defaulters ranked in the safest half vs defaulters flagged in the riskiest 20%
dd = df[df.y == 1]
missed = dd[dd.p <= df.p.median()].index
caught = dd[dd.p >= df.p.quantile(0.8)].index
cmp_cols = ["EXT_MEAN", "AGE_YEARS", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "EMPLOYED_YEARS", "AMT_INCOME_TOTAL", "BUREAU_COUNT", "PREV_COUNT", "INST_DPD_MEAN"]
mc = pd.DataFrame({"missed_defaulters_(safest_50%)": Xho.loc[missed, cmp_cols].median(), "caught_defaulters_(riskiest_20%)": Xho.loc[caught, cmp_cols].median(),
                   "all_repaid": Xho.loc[df.y == 0, cmp_cols].median()})
mc.loc["n_applicants"] = [len(missed), len(caught), int((df.y == 0).sum())]
mc.to_csv(TAB / "27_missed_vs_caught_defaulters.csv")
S["missed_defaulters_share_safest_half"] = float(len(missed) / len(dd))
fp = df[(df.y == 0) & (df.p >= df.p.quantile(0.95))]
S["false_alarm_share_of_top5pct"] = float(len(fp) / (df.p >= df.p.quantile(0.95)).sum())

# ------------------------------------------------------------------ 8. class imbalance study (LightGBM, set B, 3-fold on training split)
XB, _, _, _ = load_matrix("B")
XBtr = XB.iloc[tr].reset_index(drop=True)
rows = []
spw = float((1 - ytr.mean()) / ytr.mean())
variants = {"no adjustment": dict(), "scale_pos_weight (balanced)": dict(scale_pos_weight=spw), "is_unbalance": dict(is_unbalance=True)}
oofv = {k: np.zeros(len(ytr)) for k in list(variants) + ["undersample majority 3:1"]}
for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XBtr, ytr):
    for nm, prm in variants.items():
        p, _, _ = fit_predict(Spec(nm, "lgbm", prm), XBtr.iloc[a], ytr.iloc[a], XBtr.iloc[b])
        oofv[nm][b] = p
    rng = np.random.default_rng(SEED)
    pos = a[ytr.iloc[a].to_numpy() == 1]
    neg = rng.choice(a[ytr.iloc[a].to_numpy() == 0], size=3 * len(pos), replace=False)
    ii = np.sort(np.r_[pos, neg])
    p, _, _ = fit_predict(Spec("us", "lgbm"), XBtr.iloc[ii], ytr.iloc[ii], XBtr.iloc[b])
    oofv["undersample majority 3:1"][b] = p
for nm, pp in oofv.items():
    c = calibration_summary(ytr, pp)
    tf = best_f1_threshold(ytr, pp)
    m = threshold_metrics(ytr, pp, tf)
    rows.append(dict(variant=nm, auc=roc_auc_score(ytr, pp), avg_precision=c["avg_precision"], brier=c["brier"], mean_pred=c["mean_pred"], base_rate=c["base_rate"],
                     best_f1_threshold=tf, best_f1=m["f1"], flagged_share_at_best_f1=m["flagged_share"], recall_at_best_f1=m["recall"], precision_at_best_f1=m["precision"]))
imb = pd.DataFrame(rows)
imb.to_csv(TAB / "28_class_imbalance_study_set_B.csv", index=False)

# ------------------------------------------------------------------ 9. learning curve (tuned LightGBM, set C)
rows = []
rng = np.random.default_rng(SEED)
for frac in [0.1, 0.25, 0.5, 1.0]:
    ii = np.arange(len(ytr)) if frac == 1.0 else np.sort(StratifiedKFold(int(round(1 / frac)), shuffle=True, random_state=SEED).split(Xtr, ytr).__next__()[1])
    p, _, info = fit_predict(Spec("lc", "lgbm", TUNED), Xtr.iloc[ii], ytr.iloc[ii], Xho)
    rows.append(dict(train_fraction=frac, n_train=len(ii), holdout_auc=roc_auc_score(yho, p), best_iter=info["best_iter"]))
    print("learning curve", rows[-1], flush=True)
lc = pd.DataFrame(rows)
lc.to_csv(TAB / "29_learning_curve_tuned_set_C.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(lc["n_train"], lc["holdout_auc"], "o-")
ax.set(xscale="log", xlabel="training rows", ylabel="hold-out AUC", title="Learning curve (tuned LightGBM, all features)")
save(fig, "17_learning_curve.png")

# ------------------------------------------------------------------ 10. seeds
seed_auc = []
for sd in [1, 2, 3]:
    p, _, _ = fit_predict(Spec("seed", "lgbm", dict(TUNED, random_state=sd)), Xtr, ytr, Xho, seed=sd)
    seed_auc.append(roc_auc_score(yho, p))
S["holdout_auc_3_seeds"] = seed_auc

# ------------------------------------------------------------------ 11. final comparison table + figure
cmp_B = pd.read_csv(TAB / "10_model_comparison_set_B.csv")
abl = pd.read_csv(TAB / "12_feature_ablation_lightgbm.csv") if (TAB / "12_feature_ablation_lightgbm.csv").exists() else None
rows = [dict(model=r.model, features="B (application + engineered)", cv_auc_mean=r.cv_auc_mean, cv_auc_std=r.cv_auc_std, holdout_auc=r.holdout_auc) for r in cmp_B.itertuples()]
rows.append(dict(model="LightGBM default", features="C (all tables)", cv_auc_mean=S["cv_auc_set_C"]["default"]["mean"], cv_auc_std=S["cv_auc_set_C"]["default"]["std"], holdout_auc=S["holdout_auc"]["default"][0]))
rows.append(dict(model="LightGBM tuned", features="C (all tables)", cv_auc_mean=S["cv_auc_set_C"]["tuned"]["mean"], cv_auc_std=S["cv_auc_set_C"]["tuned"]["std"], holdout_auc=S["holdout_auc"]["tuned"][0]))
fin = pd.DataFrame(rows)
fin.to_csv(TAB / "30_final_model_comparison.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(fin.model + "  [" + fin.features.str[0] + "]", fin.holdout_auc, color=["C0"] * 5 + ["C1", "C3"])
ax.errorbar(fin.cv_auc_mean, range(len(fin)), xerr=fin.cv_auc_std, fmt="k.", label="5-fold CV mean +/- sd")
ax.set(xlim=(0.7, 0.82), xlabel="AUC", title="Model comparison: hold-out AUC (bars) and CV AUC (dots)")
ax.legend()
save(fig, "18_model_comparison.png")

# ------------------------------------------------------------------ 12. Kaggle submission: refit on ALL labelled rows
n_final = int(np.mean(iters) * 1.1)
final = lgb.LGBMClassifier(**dict(n_estimators=5000, learning_rate=0.03, subsample_freq=1, n_jobs=4, random_state=SEED, verbosity=-1,
                                  **{k: v for k, v in TUNED.items() if k != "learning_rate"}))
final.set_params(n_estimators=n_final)
final.fit(X, y)
ptest = final.predict_proba(Xtest)[:, 1]
sub = pd.DataFrame({"SK_ID_CURR": test_ids.astype(int), "TARGET": ptest})
sample = pd.read_csv("data/raw/sample_submission.csv")
assert set(sub.SK_ID_CURR) == set(sample.SK_ID_CURR) and len(sub) == len(sample)
sub.to_csv("outputs/submission.csv", index=False)
S["submission"] = dict(rows=int(len(sub)), n_trees=n_final, mean_pred_test=float(ptest.mean()), mean_pred_holdout=float(p_t.mean()), train_default_rate=float(y.mean()))
Path("outputs/final_summary.json").write_text(json.dumps(S, indent=2, default=str))
print(json.dumps(S, indent=2, default=str)[:6000])
