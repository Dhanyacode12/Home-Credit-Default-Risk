"""Cleaning and feature engineering for application_train.

Each step follows a decision in DECISIONS.md.
"""
import re

import numpy as np
import pandas as pd

EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
XNA_COLS = ["CODE_GENDER", "NAME_FAMILY_STATUS", "ORGANIZATION_TYPE"]
REDUNDANT_COLS = ["AMT_GOODS_PRICE", "CNT_CHILDREN"]   # replaced by ratios / kept via CNT_FAM_MEMBERS


def clean(df):
    """Decisions 1-2: placeholder and XNA labels."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    for col in XNA_COLS:
        df[col] = df[col].replace("XNA", np.nan)
    return df


def add_features(df, drop_redundant=False):
    """Decisions 7-9: years, ratios and combined external scores."""
    new = pd.DataFrame(index=df.index)

    new["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    new["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365
    new["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    new["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    new["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    new["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    new["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    new["ADULTS"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    new["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    new["EXT_SOURCE_MEAN"] = df[EXT_COLS].mean(axis=1)
    new["EXT_SOURCE_MIN"] = df[EXT_COLS].min(axis=1)
    new["EXT_SOURCE_PROD"] = df[EXT_COLS].prod(axis=1, skipna=False)

    out = pd.concat([df, new.replace([np.inf, -np.inf], np.nan)], axis=1)
    if drop_redundant:
        out = out.drop(columns=REDUNDANT_COLS)
    return out


def add_ext_features(df):
    """Error-analysis follow-up: more ways to use the external scores (needs add_features first).

    Missed defaulters had high external scores, so the model leans on them; these features add
    disagreement between the scores and interactions with age, employment and loan burden.
    """
    e1, e2, e3 = (df[c] for c in EXT_COLS)
    new = pd.DataFrame(index=df.index)
    new["EXT_SOURCE_MAX"] = df[EXT_COLS].max(axis=1)
    new["EXT_SOURCE_STD"] = df[EXT_COLS].std(axis=1)
    weights = pd.DataFrame({"EXT_SOURCE_1": 1.0, "EXT_SOURCE_2": 2.0, "EXT_SOURCE_3": 3.0}, index=df.index)
    weights = weights.where(df[EXT_COLS].notna())
    new["EXT_SOURCE_WEIGHTED"] = (df[EXT_COLS] * weights).sum(axis=1, min_count=1) / weights.sum(axis=1, min_count=1)
    new["EXT_1_MINUS_2"] = e1 - e2
    new["EXT_1_MINUS_3"] = e1 - e3
    new["EXT_2_MINUS_3"] = e2 - e3
    new["EXT_MEAN_X_AGE"] = df["EXT_SOURCE_MEAN"] * df["AGE_YEARS"]
    new["EXT_MEAN_X_EMPLOYED"] = df["EXT_SOURCE_MEAN"] * df["YEARS_EMPLOYED"]
    new["EXT_MEAN_X_CREDIT_INCOME"] = df["EXT_SOURCE_MEAN"] * df["CREDIT_INCOME_RATIO"]
    new["EXT_MEAN_X_ANNUITY_INCOME"] = df["EXT_SOURCE_MEAN"] * df["ANNUITY_INCOME_RATIO"]
    return pd.concat([df, new.replace([np.inf, -np.inf], np.nan)], axis=1)


def skewed_columns(X, threshold=2.0):
    """Non-negative, non-binary numeric columns with strong right skew (log1p candidates)."""
    num = X.select_dtypes("number")
    num = num.loc[:, (num.min() >= 0) & (num.nunique() > 2)]
    skew = num.skew()
    return skew[skew > threshold].index.tolist()


def log_transform(X, cols):
    """log1p on the given columns (keeps 0 valid and NaN as NaN)."""
    X = X.copy()
    X[cols] = np.log1p(X[cols])
    return X


def one_hot(X):
    """One-hot encode text columns; missing becomes its own column. Names are made LightGBM-safe."""
    text = X.select_dtypes(include=["object", "string"]).columns
    out = pd.get_dummies(X, columns=list(text), dummy_na=True, dtype="float32")
    out.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in out.columns]
    return out


def merge_rare(df, min_count=100):
    """Decision 10: merge tiny categories into 'Other'."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        counts = df[col].value_counts()
        rare = counts[counts < min_count].index
        df[col] = df[col].where(~df[col].isin(rare), "Other")
    return df
