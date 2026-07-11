# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Pipeline v1 — the original (midterm) pipeline
#
# Clean, runnable extract of the approach explored in
# [`notebook_v1.py`](notebook_v1.py). This is the pipeline that produced the **midterm submission** (`data/Group_27_Submission_v1.csv`), which scored **0.886408 AUC** on the leaderboard.
#
# Kept for reproducibility and as the baseline that v2 is measured against. Its known weakness — documented in [`notebook_v2.py`](notebook_v2.py) — is that it was tuned on a random split (0.944 CV) that did not reflect the future test window. v2 fixes that.
#
# Approach: drop the start date, collapse high-cardinality IDs to the top values, one-hot encode, then pick the best of Logistic Regression / Random Forest / XGBoost on a stratified holdout.
#
# Usage:
#
# ```bash
#     python pipeline_v1.py            # fit and write data/Group_27_Submission_v1.csv
#     python pipeline_v1.py --out PATH # write to a custom path
# ```
#

# %%
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
SUBMISSION_PATH = "data/Group_27_Submission_v1.csv"
TARGET = "Dropped_Course"
SEED = 42

COMMON_NANS = {
    "",
    "-",
    "--",
    ".",
    "?",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unknonwn",
}
ID_COLS = ["Agent_ID", "Company_ID"]
IMPORTANT_MISSING_FLAGS = [
    "Agent_ID",
    "Registration_Days_Before",
    "Physical_Course_Kits",
    "Daily_Tuition_Cost",
    "Requested_Lab_Config",
    "Payment_Terms",
]
AGENT_MIN_COUNT = 300  # clear cliff: ~15 agents, ~85% coverage
COUNTRY_MIN_COUNT = 300  # ~18 countries, ~92% coverage
COUNTRY_ALIASES = {"cn": "chn"}

# %% [markdown]
# ## 1. Cleaning and missing-value completion
#
# Numeric columns are filled with train-fitted medians, categoricals get an explicit `missing` level, and a handful of columns get missingness flags.
#

# %%
def normalize_cat_cols(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        s = df[col].astype("string").str.strip().str.lower()
        s = (
            s.str
            .replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() ]+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df[col] = s.mask(s.isin(COMMON_NANS), np.nan)
    return df


def complete_missing_values(data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    num_cols = [
        c
        for c in train_df.select_dtypes(include=["int64", "float64"]).columns
        if c not in {TARGET, "Client_ID", *ID_COLS}
    ]
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    num_medians = train_df[num_cols].median()

    df = df.drop(columns=["Client_ID"], errors="ignore")
    if "Company_ID" in df.columns:
        df["has_company_id"] = df["Company_ID"].notna().astype(int)
    for col in IMPORTANT_MISSING_FLAGS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    present_num = [c for c in num_cols if c in df.columns]
    df[present_num] = df[present_num].fillna(num_medians[present_num])

    present_cat = [c for c in cat_cols if c in df.columns]
    df = normalize_cat_cols(df, present_cat)
    df[present_cat] = df[present_cat].fillna("missing").astype("object")

    for col in ID_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .fillna("missing")
                .astype("object")
            )
    return df


# %% [markdown]
# ## 2. Dimensionality reduction and feature engineering
#
# - Drop `Course_Start_Date`: v1 chose to remove it entirely (v2 revisits this).
# - Collapse rare `Agent_ID` / `Origin_Country` values to `other`; drop raw
#   `Company_ID` (keep only the `has_company_id` flag) to avoid ~600 one-hot
#   columns.
# - Replace the two lab-config columns with a single "got what was requested"
#   flag; drop columns judged to be noise (`Lanyard_Color`, `Welcome_Gift_Type`,
#   `Returning_Client`).
# - Cap the known corrupted numeric values.
#

# %%
def keep_by_min_count(col: pd.Series, min_count: int) -> set:
    counts = col.value_counts()
    return set(counts[counts >= min_count].index)


def apply_ide_reduction(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    ref = train_df.copy()
    ref["Origin_Country"] = ref["Origin_Country"].replace(COUNTRY_ALIASES)

    countries_keep = keep_by_min_count(ref["Origin_Country"], COUNTRY_MIN_COUNT) | {
        "missing"
    }
    agents_keep = keep_by_min_count(ref["Agent_ID"], AGENT_MIN_COUNT) | {"missing"}
    df["Agent_ID"] = df["Agent_ID"].where(df["Agent_ID"].isin(agents_keep), "other")
    df["Origin_Country"] = df["Origin_Country"].where(
        df["Origin_Country"].isin(countries_keep), "other"
    )
    return df.drop(columns=["Company_ID", "Agent_ID_missing"], errors="ignore")


def apply_feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recived_requested_lab"] = (
        df["Requested_Lab_Config"] == df["Assigned_Lab_Config"]
    ).astype(int)
    return df.drop(
        columns=[
            "Assigned_Lab_Config",
            "Lanyard_Color",
            "Requested_Lab_Config",
            "Returning_Client",
            "Welcome_Gift_Type",
        ],
        errors="ignore",
    )


def apply_capping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    caps = {
        "Students_Count": (None, 3),
        "Practical_Hours": (0, 8),
        "Daily_Tuition_Cost": (None, 500),
    }
    for col, (lo, hi) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def apply_preprocessing(
    df: pd.DataFrame, train_reference: pd.DataFrame
) -> pd.DataFrame:
    train_completed = complete_missing_values(train_reference, train_reference)
    df_completed = complete_missing_values(df, train_reference)
    train_completed = train_completed.drop(
        columns=["Course_Start_Date"], errors="ignore"
    )
    df_completed = df_completed.drop(columns=["Course_Start_Date"], errors="ignore")
    df_processed = apply_ide_reduction(df_completed, train_completed)
    df_processed = apply_feature_eng(df_processed)
    return apply_capping(df_processed)


# %% [markdown]
# ## 3. Encoding and models
#
# One-hot encode the categoricals, then benchmark three models on a stratified holdout. XGBoost was the winner and is used for the final fit.
#

# %%
def encode_cats(X_train, X_predict):
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        # namespace dummies so e.g. the "missing" level of Payment_Terms does
        # not collide with the numeric Payment_Terms_missing flag column
        feature_name_combiner=lambda col, val: f"ohe__{col}__{val}",
    ).set_output(transform="pandas")
    X_train_enc = pd.concat(
        [
            X_train.drop(columns=cat_cols).reset_index(drop=True),
            enc.fit_transform(X_train[cat_cols]).reset_index(drop=True),
        ],
        axis=1,
    )
    X_predict_enc = pd.concat(
        [
            X_predict.drop(columns=cat_cols).reset_index(drop=True),
            enc.transform(X_predict[cat_cols]).reset_index(drop=True),
        ],
        axis=1,
    )
    return X_train_enc, X_predict_enc


MODELS = {
    "Logistic Regression": (lambda: LogisticRegression(max_iter=2000), True),
    "Random Forest": (
        lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
        ),
        False,
    ),
    "XGBoost": (
        lambda: XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        False,
    ),
}


def fit_predict_proba(X_train, y_train, X_predict, model, scale=False):
    X_train, X_predict = encode_cats(X_train, X_predict)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_predict = scaler.transform(X_predict)
    model.fit(X_train, y_train)
    return model.predict_proba(X_predict)[:, 1]


def benchmark(train_proc, val_proc):
    """Score all three models on the (random) validation split."""
    Xt, yt = train_proc.drop(columns=[TARGET]), train_proc[TARGET]
    Xv, yv = val_proc.drop(columns=[TARGET]), val_proc[TARGET]
    rows = []
    for name, (getter, scale) in MODELS.items():
        preds = fit_predict_proba(Xt, yt, Xv, getter(), scale)
        rows.append({"model": name, "auc": roc_auc_score(yv, preds)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


# %% [markdown]
# ## 4. Fit and write the submission
#

# %%
def run_final(out_path: str = SUBMISSION_PATH) -> pd.DataFrame:
    raw = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    tr, va = train_test_split(
        raw, test_size=0.2, random_state=SEED, stratify=raw[TARGET]
    )
    scores = benchmark(apply_preprocessing(tr, tr), apply_preprocessing(va, tr))
    print(scores.to_string(index=False))
    best = scores.iloc[0]["model"]
    print(f"\nbest on random holdout: {best}")

    train_proc = apply_preprocessing(raw, raw)
    test_proc = apply_preprocessing(test, raw)
    getter, scale = MODELS[best]
    preds = fit_predict_proba(
        train_proc.drop(columns=[TARGET]),
        train_proc[TARGET],
        test_proc,
        getter(),
        scale,
    )
    submission = pd.DataFrame({
        "Client_ID": test["Client_ID"],
        "Drop_Probability": preds,
    })
    WRITE_CSV = False
    if not WRITE_CSV:
        return submission
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(submission)} rows) using {best}")
    return submission


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the v1 model and write the submission."
    )
    parser.add_argument(
        "--out",
        default=SUBMISSION_PATH,
        help=f"output CSV (default: {SUBMISSION_PATH})",
    )
    args = parser.parse_args()
    run_final(args.out)
