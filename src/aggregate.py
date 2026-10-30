"""Aggregate each secondary table to one row per SK_ID_CURR.

Run once:  uv run python src/aggregate.py          (all tables)
           uv run python src/aggregate.py previous  (one table)
Writes data/processed/<name>.pkl (data/ is gitignored).
"""
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW.parent / "processed"
ID_COLS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
SENTINELS = {"SELLERPLACE_AREA": -1}            # "not applicable" codes that are not real values


def aggregate(df, key, prefix):
    """One row per `key`: mean/max/min/sum of numeric columns, share of each category, row count."""
    df = df.copy()
    for col in df.columns:                      # 365243 is a placeholder in DAYS_* columns
        if col.startswith("DAYS_"):
            df[col] = df[col].replace(365243, np.nan)
    for col, value in SENTINELS.items():
        if col in df.columns:
            df[col] = df[col].replace(value, np.nan)

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    dummy_cols = []
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], dtype="float32")
        dummies.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in dummies.columns]
        dummy_cols = dummies.columns.tolist()
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)

    num_cols = [c for c in df.columns if c not in ID_COLS and c not in dummy_cols]
    spec = {c: ["mean", "max", "min", "sum"] for c in num_cols}
    spec.update({c: ["mean"] for c in dummy_cols})

    grouped = df.groupby(key)
    out = grouped.agg(spec)
    out.columns = [f"{prefix}_{col}_{stat}" for col, stat in out.columns]
    out[f"{prefix}_COUNT"] = grouped.size()
    return out.astype("float32")


def read(name):
    return pd.read_csv(RAW / f"HC_{name}.csv")


def build_bureau():
    bb = aggregate(read("bureau_balance"), "SK_ID_BUREAU", "BB")
    bureau = read("bureau").merge(bb, left_on="SK_ID_BUREAU", right_index=True, how="left")
    return aggregate(bureau, "SK_ID_CURR", "BUREAU")


def build_previous():
    prev = read("previous_application")
    prev["APP_CREDIT_RATIO"] = (prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]).replace([np.inf, -np.inf], np.nan)
    return aggregate(prev, "SK_ID_CURR", "PREV")


def build_installments():
    inst = read("installments_payments")
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["PAYMENT_RATIO"] = (inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]).replace([np.inf, -np.inf], np.nan)
    return aggregate(inst, "SK_ID_CURR", "INST")


def build_installments_recent(window=365):
    """Payment behaviour in the last `window` days before the application (recent behaviour, not lifetime)."""
    cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    dtypes = {c: "float32" for c in cols[1:]} | {"SK_ID_CURR": "int32"}
    parts = []
    for chunk in pd.read_csv(RAW / "HC_installments_payments.csv", usecols=cols, dtype=dtypes, chunksize=2_000_000):
        parts.append(chunk[chunk["DAYS_INSTALMENT"] >= -window])      # keep memory small: only the window
    inst = pd.concat(parts, ignore_index=True)
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["LATE"] = (inst["DAYS_LATE"] > 0).astype("float32")
    inst["UNDERPAID"] = (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"] - 0.01).astype("float32")
    inst["PAYMENT_RATIO"] = (inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]).replace([np.inf, -np.inf], np.nan)
    g = inst.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "INSTR_DAYS_LATE_mean": g["DAYS_LATE"].mean(),
        "INSTR_DAYS_LATE_max": g["DAYS_LATE"].max(),
        "INSTR_LATE_SHARE": g["LATE"].mean(),
        "INSTR_UNDERPAID_SHARE": g["UNDERPAID"].mean(),
        "INSTR_PAYMENT_RATIO_mean": g["PAYMENT_RATIO"].mean(),
        "INSTR_COUNT": g.size(),
    })
    return out.astype("float32")


def build_pos():
    return aggregate(read("POS_CASH_balance"), "SK_ID_CURR", "POS")


def build_credit_card():
    return aggregate(read("credit_card_balance"), "SK_ID_CURR", "CC")


BUILDERS = {
    "bureau": build_bureau,
    "previous": build_previous,
    "installments": build_installments,
    "installments_recent": build_installments_recent,
    "pos": build_pos,
    "credit_card": build_credit_card,
}

if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list(BUILDERS)      # e.g. `python src/aggregate.py previous`
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        table = BUILDERS[name]()
        table.to_pickle(OUT / f"{name}.pkl")
        print(f"{name}: {table.shape[0]} applicants x {table.shape[1]} features", flush=True)
        del table
        gc.collect()
