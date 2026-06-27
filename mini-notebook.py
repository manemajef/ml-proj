# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mini modeling notebook
#
# A compact version of the final modeling path from `notebook.py`.
#
# This file intentionally avoids EDA plots and the exact `Course_Start_Date`
# one-hot feature. The official test set has mostly later start dates than the
# labeled train set, so the final preprocessing drops the exact date and keeps
# the train-fitted feature pipeline stable.
#

# %%
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def show(obj):
    try:
        display(obj)
    except NameError:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)


# %%
TARGET = "Dropped_Course"
DATE_COL = "Course_Start_Date"
TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
RANDOM_STATE = 42

# Keep this true while checking time drift. It is a stress test, not the model
# selection benchmark used for the final submission helper.
RUN_CHRONO_STRESS_TEST = True


# %% [markdown]
# ## 01 Load data and check dates
#

# %%
raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)

raw_train[DATE_COL] = pd.to_datetime(raw_train[DATE_COL], errors="coerce")
raw_test[DATE_COL] = pd.to_datetime(raw_test[DATE_COL], errors="coerce")


def date_audit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_dates = set(train_df[DATE_COL].dropna().dt.date)
    test_dates = set(test_df[DATE_COL].dropna().dt.date)
    test_seen = test_df[DATE_COL].dropna().dt.date.isin(train_dates).mean()

    return pd.DataFrame([
        {
            "data": "train",
            "rows": len(train_df),
            "min_date": train_df[DATE_COL].min(),
            "max_date": train_df[DATE_COL].max(),
            "unique_dates": train_df[DATE_COL].nunique(),
            "test_dates_seen_in_train": np.nan,
        },
        {
            "data": "test",
            "rows": len(test_df),
            "min_date": test_df[DATE_COL].min(),
            "max_date": test_df[DATE_COL].max(),
            "unique_dates": test_df[DATE_COL].nunique(),
            "test_dates_seen_in_train": test_seen,
        },
        {
            "data": "overlap",
            "rows": np.nan,
            "min_date": pd.NaT,
            "max_date": pd.NaT,
            "unique_dates": len(train_dates & test_dates),
            "test_dates_seen_in_train": test_seen,
        },
    ])


show(date_audit(raw_train, raw_test))


# %% [markdown]
# ## 02 Split helpers
#

# %%
def chronological_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    val_size: float = 0.2,
):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def make_split(df: pd.DataFrame, split_type: str = "random", val_size: float = 0.2):
    if split_type == "random":
        return train_test_split(
            df,
            test_size=val_size,
            random_state=RANDOM_STATE,
            stratify=df[TARGET],
        )

    if split_type == "chronological":
        return chronological_split(df, val_size=val_size)

    raise ValueError(f"unknown split_type: {split_type}")


# %% [markdown]
# ## 03 Missing values and categorical cleanup
#

# %%
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

ID_CATEGORICAL_COLS = ["Agent_ID", "Company_ID"]
IMPORTANT_MISSING_FLAGS = [
    "Agent_ID",
    "Registration_Days_Before",
    "Physical_Course_Kits",
    "Daily_Tuition_Cost",
    "Requested_Lab_Config",
    "Payment_Terms",
]


def normalize_cat_cols(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    df = df.copy()

    for col in cat_cols:
        s = df[col].astype("string")
        s = (
            s.str
            .strip()
            .str.lower()
            .str.replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() ]+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df[col] = s.mask(s.isin(COMMON_NANS), np.nan)

    return df


def complete_missing_values(
    data: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = data.copy()
    if train_df is None:
        train_df = data

    num_cols = [
        col
        for col in train_df.select_dtypes(include=["int64", "float64"]).columns
        if col not in {TARGET, "Client_ID", *ID_CATEGORICAL_COLS}
    ]
    cat_cols = train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_medians = train_df[num_cols].median()

    df = df.drop(columns=["Client_ID"], errors="ignore")

    if "Company_ID" in df.columns:
        df["has_company_id"] = df["Company_ID"].notna().astype(int)

    for col in IMPORTANT_MISSING_FLAGS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    present_numeric_cols = [col for col in num_cols if col in df.columns]
    df[present_numeric_cols] = df[present_numeric_cols].fillna(
        num_medians[present_numeric_cols]
    )

    present_categorical_cols = [col for col in cat_cols if col in df.columns]
    df = normalize_cat_cols(df, present_categorical_cols)
    df[present_categorical_cols] = (
        df[present_categorical_cols].fillna("missing").astype("object")
    )

    for col in ID_CATEGORICAL_COLS:
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
# ## 04 Feature engineering
#

# %%
COUNTRY_ALIASES = {"cn": "chn"}
AGENT_MIN_COUNT = 300
COUNTRY_MIN_COUNT = 300


def keep_by_min_count(col: pd.Series, min_count: int) -> set:
    counts = col.value_counts()
    return set(counts[counts >= min_count].index)


def apply_id_reduction(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    train_df = train_df.copy()

    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    train_df["Origin_Country"] = train_df["Origin_Country"].replace(COUNTRY_ALIASES)

    countries_to_keep = keep_by_min_count(
        train_df["Origin_Country"], COUNTRY_MIN_COUNT
    ) | {"missing"}
    agents_to_keep = keep_by_min_count(train_df["Agent_ID"], AGENT_MIN_COUNT) | {
        "missing"
    }

    df["Agent_ID"] = df["Agent_ID"].where(df["Agent_ID"].isin(agents_to_keep), "other")
    df["Origin_Country"] = df["Origin_Country"].where(
        df["Origin_Country"].isin(countries_to_keep), "other"
    )

    return df.drop(columns=["Company_ID", "Agent_ID_missing"], errors="ignore")


def apply_feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["received_requested_lab"] = (
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

    train_completed = train_completed.drop(columns=[DATE_COL], errors="ignore")
    df_completed = df_completed.drop(columns=[DATE_COL], errors="ignore")

    df_processed = apply_id_reduction(df_completed, train_completed)
    df_processed = apply_feature_eng(df_processed)
    df_processed = apply_capping(df_processed)

    return df_processed


# Backward-compatible alias for older notebook typo.
aplly_preprocessing = apply_preprocessing


# %% [markdown]
# ## 05 Model benchmark
#

# %%
def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def encode_cats(X_train: pd.DataFrame, X_val: pd.DataFrame):
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        feature_name_combiner=lambda col, val: f"ohe__{col}__{val}",
    ).set_output(transform="pandas")

    X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
    X_val_num = X_val.drop(columns=cat_cols).reset_index(drop=True)

    if not cat_cols:
        return X_train_num, X_val_num

    X_train_cat = enc.fit_transform(X_train[cat_cols]).reset_index(drop=True)
    X_val_cat = enc.transform(X_val[cat_cols]).reset_index(drop=True)

    return (
        pd.concat([X_train_num, X_train_cat], axis=1),
        pd.concat([X_val_num, X_val_cat], axis=1),
    )


def get_lg_model():
    return LogisticRegression(max_iter=2000)


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgb_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


MODELS = {
    "Logistic Regression": {"model_getter": get_lg_model, "scale": True},
    "Random Forest": {"model_getter": get_rf_model, "scale": False},
    "XGBoost": {"model_getter": get_xgb_model, "scale": False},
}


def score_model(train_df: pd.DataFrame, val_df: pd.DataFrame, model, scale=False):
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_train, X_val = encode_cats(X_train, X_val)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def benchmark_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, cfg in MODELS.items():
        auc = score_model(
            train_df,
            val_df,
            model=cfg["model_getter"](),
            scale=cfg["scale"],
        )
        rows.append({"model": name, "auc": auc})

    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def run_benchmark(split_type: str = "random") -> pd.DataFrame:
    train_raw, val_raw = make_split(raw_train, split_type=split_type)
    train_processed = apply_preprocessing(train_raw, train_raw)
    val_processed = apply_preprocessing(val_raw, train_raw)

    scores = benchmark_models(train_processed, val_processed)
    scores.insert(0, "split", split_type)
    return scores


validation_scores = run_benchmark("random")
show(validation_scores)


# %% [markdown]
# Chronological split is useful only as a time-drift stress test. A lower AUC
# here is expected because the validation window is later than every training
# row, which changes the distribution. It should not replace the stratified
# benchmark for choosing the final model unless the assignment explicitly asks
# for future-period validation.
#

# %%
if RUN_CHRONO_STRESS_TEST:
    chrono_scores = run_benchmark("chronological")
    show(chrono_scores)


# %% [markdown]
# ## 06 Submission helper
#

# %%
def fit_predict_proba(
    train_processed: pd.DataFrame,
    predict_processed: pd.DataFrame,
    model,
    scale: bool = False,
):
    X_train, y_train = split_xy(train_processed)
    X_predict = predict_processed.drop(columns=[TARGET], errors="ignore")
    X_train, X_predict = encode_cats(X_train, X_predict)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_predict = scaler.transform(X_predict)

    model.fit(X_train, y_train)
    return model.predict_proba(X_predict)[:, 1]


def predict_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame = raw_train,
    output_path: str = "data/Group_XX_Submission.csv",
) -> pd.DataFrame:
    train_split, val_split = make_split(train_df, split_type="random")
    processed_train = apply_preprocessing(train_split, train_split)
    processed_val = apply_preprocessing(val_split, train_split)

    score_df = benchmark_models(processed_train, processed_val)
    show(score_df)

    best = score_df.iloc[0]
    best_cfg = MODELS[best["model"]]

    processed_full_train = apply_preprocessing(train_df, train_df)
    processed_test = apply_preprocessing(test_df, train_df)

    test_preds = fit_predict_proba(
        processed_full_train,
        processed_test,
        model=best_cfg["model_getter"](),
        scale=best_cfg["scale"],
    )

    submission = pd.DataFrame({
        "Client_ID": test_df["Client_ID"],
        "Drop_Probability": test_preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"wrote {output_path} using {best['model']} (val AUC={best['auc']:.4f})")
    return submission


SAVE_SUBMISSION = False
SUBMISSION_PATH = "data/test-proba.csv"

if SAVE_SUBMISSION:
    submission = predict_test(raw_test, output_path=SUBMISSION_PATH)

