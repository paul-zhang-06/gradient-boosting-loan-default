"""EDA + data-quality + leakage diagnostics on application_train. ~3 minutes."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import read
from src.models import SEED, load_matrix
from src.plots import save

TAB = Path("outputs/tables")
TAB.mkdir(parents=True, exist_ok=True)
S = {}

raw = read("application_train")
n, p = raw.shape
y = raw["TARGET"]

# ---- target distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vc = y.value_counts()
axes[0].bar(["repaid (0)", "default (1)"], vc.values, color=["C0", "C3"])
for i, v in enumerate(vc.values):
    axes[0].text(i, v, f"{v:,}\n({v / n:.1%})", ha="center", va="bottom")
axes[0].set(title="Target distribution (class imbalance)", ylabel="applicants")
axes[1].pie(vc.values, labels=["repaid", "default"], autopct="%.1f%%", colors=["C0", "C3"])
axes[1].set_title("")
save(fig, "01_target_distribution.png")
S["n_rows"], S["n_cols"], S["default_rate"] = int(n), int(p), float(y.mean())

# ---- missing values
miss = pd.DataFrame({"missing_pct": 100 * raw.isna().mean(), "dtype": raw.dtypes.astype(str)}).sort_values("missing_pct", ascending=False)
miss.to_csv(TAB / "01_missing_value_summary.csv")
S["cols_with_missing"] = int((miss.missing_pct > 0).sum())
S["cols_over_50pct_missing"] = int((miss.missing_pct > 50).sum())
fig, ax = plt.subplots(figsize=(9, 7))
top = miss.head(30)
ax.barh(top.index[::-1], top.missing_pct[::-1])
ax.set(xlabel="% missing", title="30 columns with most missing values (application_train)")
save(fig, "02_missing_values_top30.png")

# ---- anomalies
S["days_employed_sentinel_share"] = float((raw["DAYS_EMPLOYED"] == 365243).mean())
grp = raw.assign(anom=raw["DAYS_EMPLOYED"] == 365243).groupby("anom")["TARGET"].mean()
S["default_rate_sentinel"] = {str(k): float(v) for k, v in grp.items()}
S["xna_gender"] = int((raw["CODE_GENDER"] == "XNA").sum())
S["org_xna_share"] = float((raw["ORGANIZATION_TYPE"] == "XNA").mean())

# ---- correlation heatmap of selected variables
cols = ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "REGION_RATING_CLIENT_W_CITY", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "CNT_CHILDREN"]
c = raw[cols].replace(365243, np.nan).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation of selected variables with the target and each other")
save(fig, "03_correlation_heatmap.png")
c["TARGET"].drop("TARGET").sort_values().to_csv(TAB / "02_correlation_with_target.csv")

# ---- distributions by target
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "AMT_CREDIT", "DAYS_EMPLOYED"]):
    d = raw[[col, "TARGET"]].replace(365243, np.nan).dropna()
    if col == "DAYS_BIRTH":
        d[col] = -d[col] / 365.25
    if col == "DAYS_EMPLOYED":
        d[col] = -d[col] / 365.25
    for t, lab in [(0, "repaid"), (1, "default")]:
        sns.kdeplot(d.loc[d.TARGET == t, col].sample(min(40000, (d.TARGET == t).sum()), random_state=1), ax=ax, label=lab, fill=True, alpha=.3, common_norm=False)
    ax.set_title(col + (" (years)" if col in ("DAYS_BIRTH", "DAYS_EMPLOYED") else ""))
    ax.legend()
save(fig, "04_distributions_by_target.png")

# ---- default rate by category
cat_rows = []
for col in ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]:
    g = raw.groupby(col, dropna=False)["TARGET"].agg(["mean", "size"]).reset_index()
    g.columns = ["level", "default_rate", "n"]
    g["variable"] = col
    cat_rows.append(g)
cat = pd.concat(cat_rows)
cat.to_csv(TAB / "03_default_rate_by_category.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.ravel(), ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "NAME_CONTRACT_TYPE"]):
    g = cat[(cat.variable == col) & (cat.n >= 200)].sort_values("default_rate")
    ax.barh([("(missing)" if pd.isna(v) else str(v)) for v in g.level], g.default_rate.to_numpy() * 100)
    ax.axvline(100 * y.mean(), color="r", ls="--", lw=1)
    ax.set(title=f"Default rate (%) by {col}")
    ax.tick_params(axis="y", labelsize=7)
save(fig, "05_default_rate_by_category.png")

# ---- LEAKAGE CHECK 1: single-feature AUC scan on the full feature set C (train rows only)
X, yy, Xt, ids = load_matrix("C")
rows = []
for col in X.columns:
    v = X[col]
    if not pd.api.types.is_numeric_dtype(v):
        v = v.cat.codes.astype(float).replace(-1, np.nan)
    m = v.notna().to_numpy()
    if m.sum() < 1000 or v[m].nunique() < 2:
        continue
    a = roc_auc_score(yy[m], v[m])
    rows.append(dict(feature=col, auc=a, strength=abs(a - 0.5) + 0.5, coverage=m.mean()))
sf = pd.DataFrame(rows).sort_values("strength", ascending=False)
sf.to_csv(TAB / "04_single_feature_auc_scan.csv", index=False)
S["max_single_feature_auc"] = float(sf.strength.iloc[0])
S["single_feature_top5"] = sf.head(5)[["feature", "strength"]].round(4).to_dict("records")
S["n_features_single_auc_gt_0.65"] = int((sf.strength > 0.65).sum())
fig, ax = plt.subplots(figsize=(8, 7))
t = sf.head(25).iloc[::-1]
ax.barh(t.feature, t.strength)
ax.axvline(0.7, color="r", ls="--", label="0.70: would be suspicious")
ax.set(title="Leakage scan: strongest single features (AUC, direction-free)", xlabel="AUC")
ax.legend()
ax.tick_params(axis="y", labelsize=7)
save(fig, "06_single_feature_auc_scan.png")

# SK_ID_CURR as a predictor (ordering leakage check)
raw_ids = read("application_train")["SK_ID_CURR"]
S["auc_of_SK_ID_CURR"] = float(roc_auc_score(y, raw_ids))

# ---- LEAKAGE / SHIFT CHECK 2: adversarial validation (can a model tell train rows from test rows?)
def adversarial(fs):
    Xa, _, Xat, _ = load_matrix(fs)
    Z = pd.concat([Xa, Xat], ignore_index=True)
    lab = np.r_[np.zeros(len(Xa)), np.ones(len(Xat))]
    pr, imp = np.zeros(len(lab)), 0
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Z, lab):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, colsample_bytree=0.5, n_jobs=4, verbosity=-1,
                               random_state=SEED, importance_type="gain")
        m.fit(Z.iloc[a], lab[a])
        pr[b] = m.predict_proba(Z.iloc[b])[:, 1]
        imp = imp + pd.Series(m.feature_importances_, Z.columns)
    return float(roc_auc_score(lab, pr)), (imp / imp.sum()).sort_values(ascending=False)


for fs in ("A", "B"):
    auc_adv, imp = adversarial(fs)
    S[f"adversarial_auc_set_{fs}"] = auc_adv
    imp.head(15).to_csv(TAB / f"05_adversarial_top_features_set_{fs}.csv", header=["gain_share"])
S["adversarial_top5_set_A"] = pd.read_csv(TAB / "05_adversarial_top_features_set_A.csv").head(5).round(3).to_dict("records")
te_raw = read("application_test")
S["cash_loan_share_train_vs_test"] = [float((raw["NAME_CONTRACT_TYPE"] == "Cash loans").mean()), float((te_raw["NAME_CONTRACT_TYPE"] == "Cash loans").mean())]
S["amt_credit_mean_train_vs_test"] = [float(raw["AMT_CREDIT"].mean()), float(te_raw["AMT_CREDIT"].mean())]

# ---- LEAKAGE CHECK 3: shuffled-target sanity test (a leak-free pipeline must score ~0.5)
XA, _, _, _ = load_matrix("B")
rng = np.random.default_rng(SEED)
y_shuf = pd.Series(rng.permutation(yy.to_numpy()))
oof = np.zeros(len(yy))
for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(XA, y_shuf):
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, colsample_bytree=0.5, n_jobs=4, verbosity=-1, random_state=SEED)
    m.fit(XA.iloc[a], y_shuf.iloc[a])
    oof[b] = m.predict_proba(XA.iloc[b])[:, 1]
S["shuffled_target_cv_auc"] = float(roc_auc_score(y_shuf, oof))

Path("outputs/eda_summary.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2))
