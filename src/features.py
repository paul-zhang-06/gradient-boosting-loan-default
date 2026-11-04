"""Feature engineering: ratios on the application table and SK_ID_CURR-level aggregates of the six secondary tables.

Every secondary table records the applicant's HISTORY BEFORE the current application (all time variables are negative,
relative to the application date), so aggregating them does not leak the outcome. Aggregations are unsupervised (no use of
TARGET), so they are computed once on train + test together without leakage.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .data import SENTINEL_DAYS, read


# ------------------------------------------------------------------------------------------------ application ratios
def application_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    f = pd.DataFrame(index=d.index)
    f["AGE_YEARS"] = -d["DAYS_BIRTH"] / 365.25
    f["EMPLOYED_YEARS"] = -d["DAYS_EMPLOYED"] / 365.25
    f["EMPLOYED_TO_AGE"] = d["DAYS_EMPLOYED"] / d["DAYS_BIRTH"]
    f["CREDIT_TO_INCOME"] = d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"]
    f["ANNUITY_TO_INCOME"] = d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"]
    f["CREDIT_TERM_MONTHS"] = d["AMT_CREDIT"] / d["AMT_ANNUITY"]          # implied length of the loan
    f["CREDIT_TO_GOODS"] = d["AMT_CREDIT"] / d["AMT_GOODS_PRICE"]
    f["INCOME_PER_PERSON"] = d["AMT_INCOME_TOTAL"] / d["CNT_FAM_MEMBERS"]
    f["INCOME_PER_CHILD"] = d["AMT_INCOME_TOTAL"] / (1 + d["CNT_CHILDREN"])
    f["PHONE_CHANGE_TO_AGE"] = d["DAYS_LAST_PHONE_CHANGE"] / d["DAYS_BIRTH"]
    f["REGISTRATION_TO_AGE"] = d["DAYS_REGISTRATION"] / d["DAYS_BIRTH"]
    f["CAR_AGE_TO_AGE"] = d["OWN_CAR_AGE"] / (-d["DAYS_BIRTH"] / 365.25)
    ext = d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    f["EXT_MEAN"] = ext.mean(axis=1)
    f["EXT_MIN"] = ext.min(axis=1)
    f["EXT_MAX"] = ext.max(axis=1)
    f["EXT_STD"] = ext.std(axis=1)
    f["EXT_PROD"] = ext.prod(axis=1)
    f["EXT_WEIGHTED"] = 2 * d["EXT_SOURCE_1"] + 3 * d["EXT_SOURCE_2"] + 4 * d["EXT_SOURCE_3"]
    f["EXT_N_MISSING"] = ext.isna().sum(axis=1)
    docs = [c for c in d.columns if c.startswith("FLAG_DOCUMENT_")]
    f["DOCUMENTS_PROVIDED"] = d[docs].sum(axis=1)
    f["N_ENQUIRIES_TOTAL"] = d[[c for c in d.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]].sum(axis=1, min_count=1)
    f["N_MISSING_APPLICATION"] = d.drop(columns=["TARGET"], errors="ignore").isna().sum(axis=1)   # TARGET is NaN on every test row: must not be counted
    return f.replace([np.inf, -np.inf], np.nan).astype(np.float32)


# ------------------------------------------------------------------------------------------------ aggregation helper
def _agg(df: pd.DataFrame, key: str, num_aggs: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    g = df.groupby(key).agg(num_aggs)
    g.columns = [f"{prefix}{c}_{a}".upper() for c, a in g.columns]
    return g


def _onehot_mean(df: pd.DataFrame, key: str, cols: list[str], prefix: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[cols], dummy_na=False, dtype=np.float32)
    dummies[key] = df[key].to_numpy()
    g = dummies.groupby(key).mean()
    g.columns = [f"{prefix}{c}_MEAN".upper() for c in g.columns]
    return g


def _num_cols(df, exclude):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]


STD_AGGS = ["mean", "max", "min", "sum"]


# ------------------------------------------------------------------------------------------------ one function per table
def bureau_features() -> pd.DataFrame:
    b = read("bureau")
    bb = read("bureau_balance")
    status = pd.get_dummies(bb["STATUS"], prefix="ST", dtype=np.float32)
    status["SK_ID_BUREAU"] = bb["SK_ID_BUREAU"].to_numpy()
    status["MONTHS_BALANCE"] = bb["MONTHS_BALANCE"].to_numpy()
    bbg = status.groupby("SK_ID_BUREAU").agg({**{c: ["mean"] for c in status.columns if c.startswith("ST_")}, "MONTHS_BALANCE": ["min", "max", "size"]})
    bbg.columns = [f"BB_{c}_{a}".upper() for c, a in bbg.columns]
    del bb, status
    gc.collect()
    b = b.merge(bbg, left_on="SK_ID_BUREAU", right_index=True, how="left")
    b["DEBT_TO_CREDIT"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"]
    b["OVERDUE_TO_CREDIT"] = b["AMT_CREDIT_SUM_OVERDUE"] / b["AMT_CREDIT_SUM"]
    b["ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(np.float32)
    num = _num_cols(b, ["SK_ID_CURR", "SK_ID_BUREAU"])
    out = _agg(b, "SK_ID_CURR", {c: ["mean", "max", "min", "sum"] for c in num}, "BUREAU_")
    out["BUREAU_COUNT"] = b.groupby("SK_ID_CURR").size()
    out = out.join(_onehot_mean(b, "SK_ID_CURR", ["CREDIT_ACTIVE", "CREDIT_TYPE", "CREDIT_CURRENCY"], "BUREAU_"))
    act = b[b["CREDIT_ACTIVE"] == "Active"]
    out = out.join(_agg(act, "SK_ID_CURR", {c: ["mean", "sum"] for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT"]}, "BUREAU_ACTIVE_"))
    return out.astype(np.float32)


def previous_features() -> pd.DataFrame:
    p = read("previous_application")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p[c] = p[c].replace(SENTINEL_DAYS, np.nan)
    p["APP_CREDIT_PERC"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"]
    p["CREDIT_TO_ANNUITY"] = p["AMT_CREDIT"] / p["AMT_ANNUITY"]
    num = _num_cols(p, ["SK_ID_CURR", "SK_ID_PREV"])
    out = _agg(p, "SK_ID_CURR", {c: ["mean", "max", "min"] for c in num}, "PREV_")
    out["PREV_COUNT"] = p.groupby("SK_ID_CURR").size()
    cats = [c for c in p.columns if not pd.api.types.is_numeric_dtype(p[c])]
    out = out.join(_onehot_mean(p, "SK_ID_CURR", cats, "PREV_"))
    keep = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_APPLICATION", "DAYS_DECISION", "CNT_PAYMENT", "APP_CREDIT_PERC"]
    for status, tag in [("Approved", "APPROVED"), ("Refused", "REFUSED")]:
        sub = p[p["NAME_CONTRACT_STATUS"] == status]
        out = out.join(_agg(sub, "SK_ID_CURR", {c: ["mean", "max", "min"] for c in keep}, f"PREV_{tag}_"))
    return out.astype(np.float32)


def installments_features() -> pd.DataFrame:
    i = read("installments_payments")
    i["PAYMENT_PERC"] = i["AMT_PAYMENT"] / i["AMT_INSTALMENT"]
    i["PAYMENT_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]
    i["DPD"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)           # days past due
    i["DBD"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)           # days before due
    i["LATE"] = (i["DPD"] > 0).astype(np.float32)
    i["UNDERPAID"] = (i["PAYMENT_DIFF"] > 0).astype(np.float32)
    num = _num_cols(i, ["SK_ID_CURR", "SK_ID_PREV"])
    out = _agg(i, "SK_ID_CURR", {c: ["mean", "max", "min", "sum"] for c in num}, "INST_")
    out["INST_COUNT"] = i.groupby("SK_ID_CURR").size()
    recent = i[i["DAYS_INSTALMENT"] >= -365]                                          # last year behaviour
    out = out.join(_agg(recent, "SK_ID_CURR", {c: ["mean", "max"] for c in ["DPD", "LATE", "PAYMENT_PERC", "UNDERPAID"]}, "INST_L12M_"))
    return out.astype(np.float32)


def pos_features() -> pd.DataFrame:
    p = read("POS_CASH_balance")
    p["LATE"] = (p["SK_DPD"] > 0).astype(np.float32)
    num = _num_cols(p, ["SK_ID_CURR", "SK_ID_PREV"])
    out = _agg(p, "SK_ID_CURR", {c: ["mean", "max", "min", "sum"] for c in num}, "POS_")
    out["POS_COUNT"] = p.groupby("SK_ID_CURR").size()
    out = out.join(_onehot_mean(p, "SK_ID_CURR", ["NAME_CONTRACT_STATUS"], "POS_"))
    return out.astype(np.float32)


def card_features() -> pd.DataFrame:
    c = read("credit_card_balance")
    c["BALANCE_TO_LIMIT"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"]
    c["DRAWING_TO_LIMIT"] = c["AMT_DRAWINGS_CURRENT"] / c["AMT_CREDIT_LIMIT_ACTUAL"]
    c["LATE"] = (c["SK_DPD"] > 0).astype(np.float32)
    num = _num_cols(c, ["SK_ID_CURR", "SK_ID_PREV"])
    out = _agg(c, "SK_ID_CURR", {x: ["mean", "max", "min"] for x in num}, "CARD_")
    out["CARD_COUNT"] = c.groupby("SK_ID_CURR").size()
    out = out.join(_onehot_mean(c, "SK_ID_CURR", ["NAME_CONTRACT_STATUS"], "CARD_"))
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


BUILDERS = dict(bureau=bureau_features, previous=previous_features, installments=installments_features, pos=pos_features, card=card_features)
